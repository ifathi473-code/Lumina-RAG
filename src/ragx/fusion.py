"""Fusion de classements : RRF et somme pondérée de scores normalisés."""
from __future__ import annotations

import numpy as np


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Rang (1 = meilleur) de chaque document pour chaque requête."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return ranks


def rrf(score_list: list[np.ndarray], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    weights = weights or [1.0] * len(score_list)
    out = 0
    for w, s in zip(weights, score_list):
        out = out + w / (k + rank_matrix(s))
    return out


def minmax_rows(s: np.ndarray) -> np.ndarray:
    lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - lo) / np.maximum(hi - lo, 1e-9)


def weighted_sum(score_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return sum(w * minmax_rows(s) for w, s in zip(weights, score_list))
