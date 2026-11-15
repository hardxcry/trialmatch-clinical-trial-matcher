"""Ranking metrics and paired significance testing."""
from __future__ import annotations

import numpy as np


def ndcg_at_k(gains_in_rank_order: np.ndarray, all_gains: np.ndarray, k: int = 10) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    got = np.asarray(gains_in_rank_order[:k], dtype=float)
    dcg = float(np.sum((2 ** got - 1) * disc[:len(got)]))
    ideal = np.sort(np.asarray(all_gains, dtype=float))[::-1][:k]
    idcg = float(np.sum((2 ** ideal - 1) * disc[:len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevant_in_rank_order: np.ndarray, k: int = 10) -> float:
    return float(np.mean(np.asarray(relevant_in_rank_order[:k], dtype=float))) if k else 0.0


def reciprocal_rank(relevant_in_rank_order: np.ndarray) -> float:
    hits = np.flatnonzero(relevant_in_rank_order)
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def bootstrap_mean_ci(values, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_randomization_p(a, b, n_perm: int = 20000, seed: int = 42) -> float:
    """Two-sided paired randomisation test on per-topic score differences."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n_perm + 1))
