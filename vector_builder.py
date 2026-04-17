#!/usr/bin/env python3
"""
独立脚本：从 Markdown 目录构建本地 Chroma 向量库（BGE-large-zh-v1.5）。
Embedding 设备由 knowledge_rag.get_embedding_device() 自动选择：CUDA / MPS / CPU。

用法:
  python vector_builder.py
  python vector_builder.py --md-dir ./knowledge_base --db-dir ./local_skills_db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保可导入同目录下的 knowledge_rag
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from knowledge_rag import (
    COLLECTION_NAME,
    encode_texts,
    get_embedding_device,
    get_sentence_transformer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vector_builder")


def split_markdown_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    docs = splitter.split_text(text)
    # 过滤空块
    out: list[Document] = []
    for d in docs:
        content = (d.page_content or "").strip()
        if content:
            meta = dict(d.metadata or {})
            meta["source"] = path.name
            out.append(Document(page_content=content, metadata=meta))
    return out


def build_vector_store(md_dir: Path, db_dir: Path) -> None:
    md_dir = md_dir.resolve()
    db_dir = db_dir.resolve()
    if not md_dir.is_dir():
        raise FileNotFoundError(f"Markdown 目录不存在: {md_dir}")

    device = get_embedding_device()
    logger.info("Step 0/5 设备就绪: embedding device=%s", device)

    logger.info("Step 1/5 扫描 Markdown: %s", md_dir)
    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"目录下没有 .md 文件: {md_dir}")
    logger.info("  发现 %s 个文件", len(md_files))

    logger.info("Step 2/5 按 # / ## 标题切分 (MarkdownHeaderTextSplitter)")
    all_docs: list[Document] = []
    for f in md_files:
        chunks = split_markdown_file(f)
        logger.info("  %s -> %s 个 chunk", f.name, len(chunks))
        all_docs.extend(chunks)
    if not all_docs:
        raise RuntimeError("切分后没有任何非空 chunk，请检查 Markdown 内容")

    logger.info(
        "Step 3/5 加载 SentenceTransformer（设备=%s）: %s",
        device,
        "BAAI/bge-large-zh-v1.5",
    )
    get_sentence_transformer()

    logger.info("Step 4/5 向量化 %s 个 chunk", len(all_docs))
    texts = [d.page_content for d in all_docs]
    embeddings = encode_texts(texts, is_query=False)

    logger.info("Step 5/5 写入 Chroma 持久化: %s", db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    import chromadb

    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("  已删除旧集合: %s", COLLECTION_NAME)
    except Exception:
        logger.info("  无旧集合或删除跳过，将新建: %s", COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [f"chunk_{i:06d}" for i in range(len(all_docs))]
    metadatas: list[dict] = []
    for d in all_docs:
        m = {"source": d.metadata.get("source", "")}
        h1 = d.metadata.get("Header 1")
        h2 = d.metadata.get("Header 2")
        if h1 is not None:
            m["header_1"] = str(h1)
        if h2 is not None:
            m["header_2"] = str(h2)
        metadatas.append(m)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("完成: 集合 %s 共 %s 条文档", COLLECTION_NAME, collection.count())


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地知识库向量索引")
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=_SCRIPT_DIR / "knowledge_base",
        help="Markdown 知识库目录",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=_SCRIPT_DIR / "local_skills_db",
        help="Chroma 持久化目录（默认项目下 ./local_skills_db）",
    )
    args = parser.parse_args()
    build_vector_store(args.md_dir, args.db_dir)


if __name__ == "__main__":
    main()
