"""Pipeline RAG : récupération (1er étage) -> fusion -> reranking -> sélection de contexte."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25, top_k_from_scores
from .dense import LSA, TFIDF, GloveSIF
from .features import PassageStats, pair_features
from .fusion import rank_matrix, rrf, weighted_sum
from .text import split_sentences, tokenize


class FirstStage:
    """Regroupe tous les retrievers 1er étage sur un même ensemble de passages."""

    def __init__(self, texts: list[str], glove_path: str | None = None, lsa_dim: int = 256):
        self.texts = texts
        self.bm25 = BM25().fit(texts)
        self.bm25_stem = BM25(stem=True).fit(texts)
        self.tfidf = TFIDF().fit(texts)
        self.lsa = LSA(lsa_dim).fit(texts)
        self.glove = GloveSIF(glove_path).fit(texts) if glove_path else None

    def all_scores(self, queries: list[str]) -> dict[str, np.ndarray]:
        s = {
            "bm25": self.bm25.scores(queries),
            "bm25_stem": self.bm25_stem.scores(queries),
            "tfidf": self.tfidf.scores(queries),
            "lsa": self.lsa.scores(queries),
        }
        if self.glove is not None:
            s["glove"] = self.glove.scores(queries)
            s["hybrid_rrf"] = rrf([s["bm25_stem"], s["glove"]])
            s["hybrid_rrf3"] = rrf([s["bm25_stem"], s["glove"], s["lsa"]])
        return s


def candidates_and_features(queries: list[str], scores: dict[str, np.ndarray], first_key: str, stats: PassageStats, K: int = 50):
    """Retourne (cand[nq,K], liste de matrices de caractéristiques)."""
    cand, _ = top_k_from_scores(scores[first_key], K)
    rk = {k: rank_matrix(scores[k]) for k in ("bm25_stem", "glove", "lsa")}
    feats = []
    for i, q in enumerate(queries):
        c = cand[i]
        base = {
            "bm25": scores["bm25_stem"][i, c], "glove": scores["glove"][i, c],
            "lsa": scores["lsa"][i, c], "tfidf": scores["tfidf"][i, c],
            "rank_bm25": np.log1p(rk["bm25_stem"][i, c]), "rank_glove": np.log1p(rk["glove"][i, c]),
            "rank_lsa": np.log1p(rk["lsa"][i, c]), "rrf": scores["hybrid_rrf"][i, c],
        }
        feats.append(pair_features(q, tokenize(q, stem=True), c, stats, base))
    return cand, feats


def select_answer_sentence(query: str, passages: list[str], idf: dict[str, float] | None = None):
    """Lecteur extractif minimal : phrase du contexte la plus proche de la question."""
    qset = set(tokenize(query, stem=True))
    best, best_s, best_p = "", -1.0, -1
    for pi, p in enumerate(passages):
        for s in split_sentences(p):
            st = set(tokenize(s, stem=True))
            sc = sum((idf or {}).get(t, 1.0) for t in qset & st)
            if sc > best_s:
                best, best_s, best_p = s, sc, pi
    return best, best_p


@dataclass
class StageTimes:
    first: float = 0.0
    rerank: float = 0.0
    read: float = 0.0


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - t) * 1000.0
