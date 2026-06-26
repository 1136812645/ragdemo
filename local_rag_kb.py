#!/usr/bin/env python3
import argparse
import json
import math
import re
import sqlite3
import sys
import textwrap
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DOCS_DIR = Path("file")
DEFAULT_DB_PATH = Path(".rag_store") / "knowledge_base.sqlite3"
DEFAULT_CHAT_MODEL = "qwen3.6:latest"
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "what", "when", "who", "how",
    "why", "which", "that", "this", "those", "these", "it", "as", "by", "from",
    "first", "into", "their", "them", "then", "than", "about", "after", "before",
}


@dataclass
class ChunkRecord:
    document_path: str
    title: str
    chunk_index: int
    text: str


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "无法连接到 Ollama。请确认 `ollama serve` 正在运行，且监听在 127.0.0.1:11434。"
        ) from error


def embed_texts(base_url: str, model: str, texts: Sequence[str], timeout: float) -> list[list[float]]:
    response = post_json(
        f"{base_url}/api/embed",
        {"model": model, "input": list(texts)},
        timeout,
    )
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError(f"Embedding 响应异常: {response}")
    return embeddings


def generate_text(base_url: str, model: str, prompt: str, timeout: float) -> str:
    response = post_json(
        f"{base_url}/api/generate",
        {
            "model": model,
            "stream": False,
            "think": False,
            "prompt": prompt,
        },
        timeout,
    )
    try:
        content = response.get("response", "")
        if content and content.strip():
            return content.strip()
        raise RuntimeError(f"Generate 响应为空: {response}")
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Generate 响应异常: {response}") from error


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(document_id, chunk_index)
        );
        """
    )
    connection.commit()


def normalize_whitespace(text: str) -> str:
    text = unescape(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_xml_text(xhtml: bytes) -> str:
    root = ET.fromstring(xhtml)
    pieces = []
    for text in root.itertext():
        cleaned = normalize_whitespace(text)
        if cleaned:
            pieces.append(cleaned)
    return "\n".join(pieces)


def read_epub_text(epub_path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(epub_path) as archive:
        container_root = ET.fromstring(archive.read("META-INF/container.xml"))
        namespace = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        rootfile = container_root.find("c:rootfiles/c:rootfile", namespace)
        if rootfile is None:
            raise RuntimeError(f"无法在 {epub_path} 中找到 OPF 清单")

        opf_path = rootfile.attrib["full-path"]
        opf_dir = Path(opf_path).parent
        opf_root = ET.fromstring(archive.read(opf_path))
        opf_ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}

        title = opf_root.findtext("opf:metadata/dc:title", default=epub_path.stem, namespaces=opf_ns)
        manifest = {}
        for item in opf_root.findall("opf:manifest/opf:item", opf_ns):
            item_id = item.attrib.get("id")
            href = item.attrib.get("href")
            media_type = item.attrib.get("media-type", "")
            if item_id and href:
                manifest[item_id] = (href, media_type)

        text_parts = []
        for itemref in opf_root.findall("opf:spine/opf:itemref", opf_ns):
            item_id = itemref.attrib.get("idref")
            if not item_id or item_id not in manifest:
                continue
            href, media_type = manifest[item_id]
            if "xhtml" not in media_type and not href.endswith((".xhtml", ".html", ".htm")):
                continue
            entry_path = str((opf_dir / href).as_posix())
            text = strip_xml_text(archive.read(entry_path))
            if text:
                text_parts.append(text)

    return normalize_whitespace(title), "\n\n".join(text_parts)


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    compact = normalize_whitespace(text)
    if not compact:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [compact]

    chunks = []
    current_parts = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = normalize_whitespace(paragraph)
        if not paragraph:
            continue

        additional = len(paragraph) + (1 if current_parts else 0)
        if current_parts and current_length + additional > chunk_size:
            chunks.append(" ".join(current_parts))
            overlap_text = chunks[-1][-overlap:].strip() if overlap > 0 else ""
            current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_length = sum(len(part) for part in current_parts) + max(len(current_parts) - 1, 0)
            continue

        current_parts.append(paragraph)
        current_length += additional

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [chunk for chunk in chunks if chunk.strip()]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z']+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def lexical_score(query: str, content: str, title: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    content_lower = content.lower()
    title_lower = title.lower()
    unique_tokens = set(query_tokens)
    overlap = sum(1 for token in unique_tokens if token in content_lower)
    title_hits = sum(1 for token in unique_tokens if token in title_lower)
    phrase_bonus = 1.0 if normalize_whitespace(query).lower() in content_lower else 0.0
    return (overlap / len(unique_tokens)) + (0.2 * title_hits) + (0.3 * phrase_bonus)


def iter_epub_files(docs_dir: Path) -> Iterable[Path]:
    yield from sorted(docs_dir.glob("*.epub"))


def upsert_document(
    connection: sqlite3.Connection,
    document_path: Path,
    title: str,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> None:
    updated_at = document_path.stat().st_mtime
    connection.execute(
        "INSERT INTO documents(path, title, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at",
        (document_path.as_posix(), title, updated_at),
    )
    document_id = connection.execute(
        "SELECT id FROM documents WHERE path = ?",
        (document_path.as_posix(),),
    ).fetchone()[0]
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    connection.executemany(
        "INSERT INTO chunks(document_id, chunk_index, content, embedding) VALUES(?, ?, ?, ?)",
        [
            (document_id, index, chunk, json.dumps(vector))
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=False))
        ],
    )


def needs_reindex(connection: sqlite3.Connection, document_path: Path) -> bool:
    row = connection.execute(
        "SELECT updated_at FROM documents WHERE path = ?",
        (document_path.as_posix(),),
    ).fetchone()
    if row is None:
        return True
    return float(row[0]) < document_path.stat().st_mtime


def build_index(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        init_db(connection)

        indexed = 0
        skipped = 0
        for epub_path in iter_epub_files(docs_dir):
            if not args.force and not needs_reindex(connection, epub_path):
                skipped += 1
                continue

            print(f"开始索引: {epub_path.name}", flush=True)
            title, text = read_epub_text(epub_path)
            chunks = split_into_chunks(text, args.chunk_size, args.overlap)
            if not chunks:
                print(f"跳过空文档: {epub_path}", flush=True)
                continue

            embeddings = []
            for batch_number, start in enumerate(range(0, len(chunks), args.batch_size), start=1):
                batch = chunks[start : start + args.batch_size]
                embeddings.extend(embed_texts(args.ollama_url, args.embed_model, batch, args.timeout))
                current = min(start + len(batch), len(chunks))
                if batch_number == 1 or current == len(chunks) or batch_number % 10 == 0:
                    print(
                        f"  embedding 进度: {current}/{len(chunks)}",
                        flush=True,
                    )

            upsert_document(connection, epub_path, title, chunks, embeddings)
            connection.commit()
            indexed += 1
            print(
                f"已索引: {epub_path.name} | 标题: {title} | 分块: {len(chunks)}",
                flush=True,
            )

    print(f"完成。新增/更新 {indexed} 个文档，跳过 {skipped} 个文档。", flush=True)
    return 0


def load_all_chunks(connection: sqlite3.Connection) -> list[tuple[str, str, int, str, list[float]]]:
    rows = connection.execute(
        """
        SELECT documents.path, documents.title, chunks.chunk_index, chunks.content, chunks.embedding
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        ORDER BY documents.path, chunks.chunk_index
        """
    ).fetchall()
    return [
        (row[0], row[1], int(row[2]), row[3], json.loads(row[4]))
        for row in rows
    ]


def search_chunks(
    connection: sqlite3.Connection,
    query: str,
    ollama_url: str,
    embed_model: str,
    top_k: int,
    timeout: float,
) -> list[tuple[float, ChunkRecord]]:
    chunks = load_all_chunks(connection)
    if not chunks:
        raise RuntimeError("知识库为空，请先执行 index。")

    query_vector = embed_texts(ollama_url, embed_model, [query], timeout)[0]
    scored = []
    for path, title, chunk_index, content, vector in chunks:
        semantic = cosine_similarity(query_vector, vector)
        lexical = lexical_score(query, content, title)
        score = (semantic * 0.75) + (lexical * 0.25)
        scored.append(
            (
                score,
                ChunkRecord(
                    document_path=path,
                    title=title,
                    chunk_index=chunk_index,
                    text=content,
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def run_search(args: argparse.Namespace) -> int:
    with sqlite3.connect(args.db_path) as connection:
        results = search_chunks(
            connection,
            query=args.query,
            ollama_url=args.ollama_url,
            embed_model=args.embed_model,
            top_k=args.top_k,
            timeout=args.timeout,
        )

    for rank, (score, chunk) in enumerate(results, start=1):
        preview = textwrap.shorten(chunk.text, width=280, placeholder="...")
        print(f"[{rank}] score={score:.4f}")
        print(f"文档: {chunk.title}")
        print(f"路径: {chunk.document_path}")
        print(f"分块: {chunk.chunk_index}")
        print(preview)
        print()
    return 0


def run_ask(args: argparse.Namespace) -> int:
    with sqlite3.connect(args.db_path) as connection:
        results = search_chunks(
            connection,
            query=args.question,
            ollama_url=args.ollama_url,
            embed_model=args.embed_model,
            top_k=args.top_k,
            timeout=args.timeout,
        )

    context_blocks = []
    for score, chunk in results:
        context_blocks.append(
            f"[title] {chunk.title}\n[path] {chunk.document_path}\n[chunk] {chunk.chunk_index}\n[score] {score:.4f}\n[text]\n{chunk.text[:1200]}"
        )

    prompt = (
        "Answer only from the provided book excerpts. "
        "If the excerpts are insufficient, say you do not know. "
        "Use at most 3 sentences and do not include preambles.\n\n"
        f"Question: {args.question}\n\n"
        "Excerpts:\n"
        + "\n\n---\n\n".join(context_blocks)
    )
    answer = generate_text(args.ollama_url, args.chat_model, prompt, args.timeout)
    print(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 EPUB 向量知识库与查询工具")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="文档目录")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite 索引文件路径")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama 服务地址")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding 模型")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="聊天模型")
    parser.add_argument("--timeout", type=float, default=180, help="请求超时时间（秒）")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="构建或更新本地向量索引")
    index_parser.add_argument("--chunk-size", type=int, default=4000, help="文本分块大小")
    index_parser.add_argument("--overlap", type=int, default=300, help="相邻分块重叠字符数")
    index_parser.add_argument("--batch-size", type=int, default=16, help="embedding 批大小")
    index_parser.add_argument("--force", action="store_true", help="强制重建全部文档索引")
    index_parser.set_defaults(func=build_index)

    search_parser = subparsers.add_parser("search", help="搜索最相关的文档分块")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("--top-k", type=int, default=5, help="返回结果数")
    search_parser.set_defaults(func=run_search)

    ask_parser = subparsers.add_parser("ask", help="基于检索结果向模型提问")
    ask_parser.add_argument("question", help="问题文本")
    ask_parser.add_argument("--top-k", type=int, default=4, help="提供给模型的上下文块数")
    ask_parser.set_defaults(func=run_ask)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())