"""BM25 (Okapi) implémenté avec des matrices creuses : rapide, sans dépendance externe."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, stem: bool = False, stop: bool = True):
        self.k1, self.b, self.stem, self.stop = k1, b, stem, stop

    def _an(self, t: str):
        return tokenize(t, stop=self.stop, stem=self.stem)

    def fit(self, docs: list[str]) -> "BM25":
        self.cv = CountVectorizer(analyzer=self._an)
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        dl = np.asarray(tf.sum(1)).ravel()
        self.avgdl = dl.mean()
        df = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / self.avgdl)
        w = self.idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)   # docs x termes
        self.WT = self.W.T.tocsr()
        return self

    def scores(self, queries: list[str]) -> np.ndarray:
        q = self.cv.transform(queries)
        q.data[:] = 1.0                      # présence du terme (requêtes courtes)
        return (q @ self.WT).toarray()       # (nq, nd)

    def topk(self, queries: list[str], k: int = 100):
        s = self.scores(queries)
        return top_k_from_scores(s, k)


def top_k_from_scores(s: np.ndarray, k: int):
    k = min(k, s.shape[1])
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(s, idx, 1)
    o = np.argsort(-part, axis=1, kind="stable")
    return np.take_along_axis(idx, o, 1), np.take_along_axis(part, o, 1)
