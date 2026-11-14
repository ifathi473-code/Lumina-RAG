"""Récupération « dense » : GloVe pondéré par l'IDF (+ suppression de la composante
commune, SIF) et LSA (TF-IDF + SVD). Retrievers neuronaux optionnels : voir neural.py."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import tokenize


def load_glove(path: str | Path, vocab: set[str] | None = None):
    vecs, words = [], []
    with gzip.open(path, "rt", encoding="utf8") as f:
        next(f)  # en-tête word2vec
        for line in f:
            w, rest = line.split(" ", 1)
            if vocab is not None and w not in vocab:
                continue
            words.append(w)
            vecs.append(np.asarray(rest.split(), dtype=np.float32))
    return {w: i for i, w in enumerate(words)}, np.vstack(vecs)


class GloveSIF:
    """Embedding de texte = moyenne des vecteurs de mots pondérés par idf, puis retrait
    de la 1re composante principale (Arora et al., 2017)."""

    def __init__(self, glove_path: str | Path, remove_pc: bool = True):
        self.glove_path, self.remove_pc = glove_path, remove_pc

    def fit(self, docs: list[str]) -> "GloveSIF":
        toks = [tokenize(d) for d in docs]
        vocab = {t for ts in toks for t in ts}
        # on garde aussi les mots des questions fréquentes : vocabulaire GloVe complet limité au corpus
        self.w2i, self.E = load_glove(self.glove_path, vocab | _COMMON_Q)
        n = len(docs)
        df: dict[str, int] = {}
        for ts in toks:
            for t in set(ts):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: np.log(1 + n / c) for t, c in df.items()}
        self.default_idf = float(np.log(1 + n))
        M = np.vstack([self._raw(ts) for ts in toks])
        if self.remove_pc:
            u = TruncatedSVD(1, random_state=0).fit(M).components_[0]
            self.pc = u.astype(np.float32)
        else:
            self.pc = None
        self.D = self._post(M)
        return self

    def _raw(self, ts: list[str]) -> np.ndarray:
        v = np.zeros(self.E.shape[1], dtype=np.float32)
        wsum = 0.0
        for t in ts:
            i = self.w2i.get(t)
            if i is None:
                continue
            w = self.idf.get(t, self.default_idf)
            v += w * self.E[i]
            wsum += w
        return v / wsum if wsum else v

    def _post(self, M: np.ndarray) -> np.ndarray:
        if self.pc is not None:
            M = M - np.outer(M @ self.pc, self.pc)
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, 1e-9)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._post(np.vstack([self._raw(tokenize(t)) for t in texts]))

    def scores(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries) @ self.D.T


_COMMON_Q = {"who", "when", "where", "why", "how", "which", "what"}


class LSA:
    """TF-IDF -> SVD tronquée (analyse sémantique latente)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def fit(self, docs: list[str]) -> "LSA":
        self.tv = TfidfVectorizer(analyzer=lambda t: tokenize(t, stem=True), sublinear_tf=True)
        X = self.tv.fit_transform(docs)
        self.svd = TruncatedSVD(max(1, min(self.dim, X.shape[1] - 1, X.shape[0] - 1)), random_state=0).fit(X)
        self.D = self._norm(self.svd.transform(X))
        return self

    @staticmethod
    def _norm(M):
        return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

    def scores(self, queries: list[str]) -> np.ndarray:
        q = self._norm(self.svd.transform(self.tv.transform(queries)))
        return q @ self.D.T


class TFIDF:
    def fit(self, docs: list[str]) -> "TFIDF":
        self.tv = TfidfVectorizer(analyzer=lambda t: tokenize(t, stem=True), sublinear_tf=True)
        self.X = self.tv.fit_transform(docs)
        return self

    def scores(self, queries: list[str]) -> np.ndarray:
        return (self.tv.transform(queries) @ self.X.T).toarray()
