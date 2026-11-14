"""Extraction de caractéristiques (requête, passage) pour le reranker appris."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .text import split_sentences, tokenize

QTYPES = ["what", "who", "when", "where", "why", "how", "which", "other"]
FEATURE_NAMES = [
    "bm25", "bm25_rel", "glove_cos", "lsa_cos", "tfidf_cos",
    "rank_bm25", "rank_glove", "rank_lsa", "rrf",
    "coverage", "idf_coverage", "bigram_match", "bigram_frac", "proximity",
    "first_match_pos", "best_sent_cov", "len_passage", "len_query",
    "n_numbers", "n_years", "n_caps",
] + [f"qtype_{t}" for t in QTYPES]

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_CAP = re.compile(r"\b[A-Z][a-z]+")


def qtype(q: str) -> int:
    ql = q.lower()
    best, pos = len(QTYPES) - 1, 10**9
    for i, t in enumerate(QTYPES[:-1]):
        m = re.search(rf"\b{t}\b", ql)
        if m and m.start() < pos:
            best, pos = i, m.start()
    return best


class PassageStats:
    """Pré-calcul par passage (tokens racinisés, positions, phrases, compteurs)."""

    def __init__(self, texts: list[str]):
        self.tok = [tokenize(t, stem=True) for t in texts]
        self.pos: list[dict[str, list[int]]] = []
        self.bigrams: list[set[tuple[str, str]]] = []
        for ts in self.tok:
            d: dict[str, list[int]] = {}
            for i, t in enumerate(ts):
                d.setdefault(t, []).append(i)
            self.pos.append(d)
            self.bigrams.append(set(zip(ts, ts[1:])))
        self.sents = [[set(tokenize(s, stem=True)) for s in split_sentences(t)] or [set()] for t in texts]
        self.n_years = np.array([len(_YEAR.findall(t)) for t in texts], dtype=np.float32)
        self.n_nums = np.array([len(_NUM.findall(t)) for t in texts], dtype=np.float32)
        self.n_caps = np.array([len(_CAP.findall(t)) for t in texts], dtype=np.float32)
        n = len(texts)
        df = Counter(t for d in self.pos for t in d)
        self.idf = {t: float(np.log(1 + n / c)) for t, c in df.items()}
        self.default_idf = float(np.log(1 + n))


def _min_window(pos_lists: list[list[int]]) -> int:
    pairs = sorted((p, j) for j, pl in enumerate(pos_lists) for p in pl)
    need, have, cnt, best, l = len(pos_lists), [0] * len(pos_lists), 0, 10**9, 0
    for r, (pr, jr) in enumerate(pairs):
        if have[jr] == 0:
            cnt += 1
        have[jr] += 1
        while cnt == need:
            best = min(best, pr - pairs[l][0] + 1)
            jl = pairs[l][1]
            have[jl] -= 1
            if have[jl] == 0:
                cnt -= 1
            l += 1
    return best


def pair_features(q: str, qtoks: list[str], cand: np.ndarray, stats: PassageStats, base: dict[str, np.ndarray]) -> np.ndarray:
    """base : scores/rangs de 1er étage alignés sur `cand` (bm25, glove, lsa, tfidf, rank_*, rrf)."""
    qset = list(dict.fromkeys(qtoks))
    qbig = set(zip(qtoks, qtoks[1:]))
    qidf_sum = sum(stats.idf.get(t, stats.default_idf) for t in qset) or 1.0
    qt = qtype(q)
    bm = base["bm25"]
    rows = np.zeros((len(cand), len(FEATURE_NAMES)), dtype=np.float32)
    for r, pid in enumerate(cand):
        pos = stats.pos[pid]
        matched = [t for t in qset if t in pos]
        cov = len(matched) / max(len(qset), 1)
        idf_cov = sum(stats.idf.get(t, stats.default_idf) for t in matched) / qidf_sum
        bg = len(qbig & stats.bigrams[pid])
        if len(matched) >= 2:
            prox = len(matched) / _min_window([pos[t] for t in matched])
        else:
            prox = 1.0 if matched else 0.0
        first = min((pos[t][0] for t in matched), default=-1)
        plen = max(len(stats.tok[pid]), 1)
        best_sent = max((len(qs & s) for s in stats.sents[pid] for qs in [set(qset)]), default=0) / max(len(qset), 1)
        row = [
            bm[r], bm[r] / max(bm.max(), 1e-9), base["glove"][r], base["lsa"][r], base["tfidf"][r],
            base["rank_bm25"][r], base["rank_glove"][r], base["rank_lsa"][r], base["rrf"][r],
            cov, idf_cov, bg, bg / max(len(qbig), 1), prox,
            (first / plen) if first >= 0 else 1.0, best_sent, np.log1p(plen), len(qtoks),
            stats.n_nums[pid], stats.n_years[pid], stats.n_caps[pid],
        ] + [1.0 if i == qt else 0.0 for i in range(len(QTYPES))]
        rows[r] = row
    return rows
