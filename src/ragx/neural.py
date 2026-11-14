"""Briques NEURONALES optionnelles (nécessitent un accès à Hugging Face).

Non exécutées dans l'environnement qui a produit les résultats principaux du rapport
(accès réseau limité) ; à lancer avec scripts/05_neural_optional.py sur votre machine."""
from __future__ import annotations

import numpy as np


class BiEncoder:
    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", batch: int = 64):
        from sentence_transformers import SentenceTransformer
        self.m, self.batch = SentenceTransformer(name), batch
        self.name = name

    def fit(self, docs: list[str]) -> "BiEncoder":
        self.D = self.m.encode(docs, batch_size=self.batch, normalize_embeddings=True, show_progress_bar=True)
        return self

    def scores(self, queries: list[str]) -> np.ndarray:
        q = self.m.encode(queries, batch_size=self.batch, normalize_embeddings=True)
        return q @ self.D.T


class CrossEncoderReranker:
    def __init__(self, name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch: int = 64):
        from sentence_transformers import CrossEncoder
        self.m, self.batch, self.name = CrossEncoder(name), batch, name

    def rerank(self, query: str, passages: list[str]) -> np.ndarray:
        return np.asarray(self.m.predict([(query, p) for p in passages], batch_size=self.batch))
