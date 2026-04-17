"""
RAG 共用逻辑：BGE-large-zh-v1.5 + ChromaDB。
Embedding 设备自适应：CUDA（优先）→ MPS（Apple Silicon）→ CPU。
供 vector_builder.py 与 handler.py 复用。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# 项目内模型缓存（须在导入 huggingface / sentence_transformers 之前设置）
_PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = _PROJECT_ROOT / "local_models"
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
_local_models_abs = str(LOCAL_MODELS_DIR.resolve())
os.environ["HF_HOME"] = _local_models_abs
os.environ["HUGGINGFACE_HUB_CACHE"] = str((LOCAL_MODELS_DIR / "hub").resolve())
os.environ["TRANSFORMERS_CACHE"] = str((LOCAL_MODELS_DIR / "transformers").resolve())
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str((LOCAL_MODELS_DIR / "sentence_transformers").resolve())

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL_ID = "BAAI/bge-large-zh-v1.5"
COLLECTION_NAME = "sales_knowledge"
# BGE 中文检索推荐查询前缀（官方说明）
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
# 查询侧过长文本截断，避免远超模型 max length（约 512 tokens）
MAX_QUERY_CHARS = 2000


@lru_cache(maxsize=1)
def get_embedding_device() -> str:
    """
    动态选择句向量计算设备（整进程内缓存一次）：
    1) Windows/Linux NVIDIA：torch.cuda.is_available() → cuda
    2) Apple Silicon：torch.backends.mps.is_available() → mps
    3) 兜底：cpu
    """
    import torch

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA"
        logger.info("Embedding 使用 CUDA: %s", name)
        print(f"[knowledge_rag] Embedding 硬件: CUDA ({name})", flush=True)
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Embedding 使用 MPS (Apple Silicon Metal)")
        print("[knowledge_rag] Embedding 硬件: MPS (Apple Silicon)", flush=True)
        return "mps"
    logger.warning("Embedding 无 CUDA/MPS，使用 CPU（速度较慢）")
    print("[knowledge_rag] Embedding 硬件: CPU（未检测到 CUDA/MPS）", flush=True)
    return "cpu"


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    device = get_embedding_device()
    cache_folder = str(LOCAL_MODELS_DIR.resolve())
    logger.info(
        "加载 Embedding 模型: %s, device=%s, cache_folder=%s",
        EMBED_MODEL_ID,
        device,
        cache_folder,
    )
    # cache_folder + HF_HOME：权重落在 ./local_models；已缓存则离线加载，不重复下载
    model = SentenceTransformer(
        EMBED_MODEL_ID,
        device=device,
        cache_folder=cache_folder,
    )
    return model


def encode_texts(texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
    """批量编码；查询文本自动加 BGE 中文前缀。"""
    model = get_sentence_transformer()
    to_encode: List[str] = []
    for t in texts:
        t = (t or "").strip()
        if is_query:
            to_encode.append(BGE_ZH_QUERY_PREFIX + t)
        else:
            to_encode.append(t)
    emb = model.encode(
        to_encode,
        normalize_embeddings=True,
        show_progress_bar=len(to_encode) > 8,
    )
    return emb.tolist()


def retrieve_top_k(
    transcript: str,
    db_path: Path,
    top_k: int = 3,
) -> Tuple[List[str], List[dict]]:
    """
    用语义检索返回 top_k 个知识块正文与 metadata。
    """
    q = (transcript or "").strip()
    if not q:
        logger.warning("转录为空，跳过检索")
        return [], []
    q = q[:MAX_QUERY_CHARS]
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        logger.warning("向量库目录不存在: %s（请先运行 vector_builder.py）", db_path)
        return [], []
    logger.info("连接 Chroma: %s", db_path)
    try:
        collection = _get_chroma_collection_cached(str(db_path))
    except Exception as e:
        logger.warning("无法打开 Chroma 集合 %s: %s", COLLECTION_NAME, e)
        return [], []
    q_emb = encode_texts([q], is_query=True)
    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs: List[str] = []
    metas: List[dict] = []
    if res["documents"] and res["documents"][0]:
        for i, doc in enumerate(res["documents"][0]):
            if doc:
                docs.append(doc)
                meta = {}
                if res.get("metadatas") and res["metadatas"][0] and i < len(res["metadatas"][0]):
                    meta = res["metadatas"][0][i] or {}
                metas.append(meta)
    logger.info("RAG 检索命中 %s 条（请求 top_k=%s）", len(docs), top_k)
    return docs, metas


@lru_cache(maxsize=4)
def _get_chroma_collection_cached(db_path_str: str):
    """缓存 Chroma collection，避免每次检索重复初始化客户端。"""
    client = chromadb.PersistentClient(path=db_path_str)
    return client.get_collection(name=COLLECTION_NAME)


def warmup_rag(db_path: Path):
    """
    预热 RAG：
    1) 预加载 embedding 模型
    2) 尝试打开 Chroma collection（失败仅告警）
    """
    get_sentence_transformer()
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        logger.warning("warmup: 向量库目录不存在，跳过 Chroma 预热: %s", db_path)
        return
    try:
        _get_chroma_collection_cached(str(db_path))
        logger.info("warmup: Chroma collection 就绪")
    except Exception as e:
        logger.warning("warmup: Chroma 预热失败: %s", e)


def format_reference_blocks(chunks: List[str], metas: Optional[List[dict]] = None) -> str:
    """拼成 Prompt 中的 Reference Context。"""
    if not chunks:
        return "（知识库未返回相关片段；请先运行 vector_builder.py 构建索引。）"
    parts: List[str] = []
    for i, text in enumerate(chunks, 1):
        header = f"### 参考片段 {i}"
        if metas and i - 1 < len(metas) and metas[i - 1]:
            src = metas[i - 1].get("source") or metas[i - 1].get("file") or ""
            if src:
                header += f"（来源: {src}）"
        parts.append(f"{header}\n\n{text.strip()}")
    return "\n\n---\n\n".join(parts)
