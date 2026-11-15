"""Four ranking strategies over a pool of trial passages:
BM25 (lexical), dense bi-encoder, hybrid reciprocal-rank fusion, cross-encoder re-ranking."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def dense_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(DENSE_MODEL, device=device())
    m.max_seq_length = 512
    return m


@lru_cache(maxsize=1)
def rerank_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device=device(), max_length=512)


def embed_passages(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    return dense_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=show_progress).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    return dense_model().encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0].astype(np.float32)


def bm25_scores(query: str, tokenized_docs: list[list[str]]) -> np.ndarray:
    return np.asarray(BM25Okapi(tokenized_docs).get_scores(tokenize(query)))


def dense_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    return doc_vecs @ query_vec


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion (Cormack et al. 2009): sum of 1 / (k + rank)."""
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s), dtype=int)
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        fused += 1.0 / (k + ranks)
    return fused


def rerank(query: str, texts: list[str], base_scores: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Re-score the top_k candidates with a cross-encoder; the rest keep their order below."""
    order = np.argsort(-base_scores, kind="stable")
    head = order[:top_k]
    ce = rerank_model().predict([(query, texts[i]) for i in head], batch_size=32,
                                show_progress_bar=False)
    out = np.full(len(texts), -np.inf)
    out[head] = ce
    tail = order[top_k:]
    out[tail] = np.min(ce) - 1 - np.arange(len(tail))
    return out
