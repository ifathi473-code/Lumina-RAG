"""Service RAG de bout en bout (utilisé par l'application Streamlit et l'API FastAPI)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .bm25 import top_k_from_scores
from .features import PassageStats
from .fusion import rrf
from .pipeline import FirstStage, candidates_and_features, select_answer_sentence

MODES = ["BM25", "Hybride (RRF)", "Hybride + reranker LightGBM"]


@dataclass
class SearchResult:
    passages: list[dict]
    answer_sentence: str
    timings_ms: dict[str, float] = field(default_factory=dict)


class RagService:
    def __init__(self, texts: list[str], glove_path: str | Path | None, model_path: str | Path | None, K: int = 50):
        self.texts = texts
        self.K = min(K, len(texts))
        self.fs = FirstStage(texts, str(glove_path) if glove_path and Path(glove_path).exists() else None)
        self.stats = PassageStats(texts)
        self.model = lgb.Booster(model_file=str(model_path)) if model_path and Path(model_path).exists() else None

    @property
    def modes(self) -> list[str]:
        out = ["BM25"]
        if self.fs.glove is not None:
            out.append("Hybride (RRF)")
            if self.model is not None:
                out.append("Hybride + reranker LightGBM")
        return out

    def search(self, query: str, mode: str = "BM25", top: int = 5, K: int | None = None) -> SearchResult:
        t = {}
        t0 = time.perf_counter()
        S = {"bm25_stem": self.fs.bm25_stem.scores([query]), "lsa": self.fs.lsa.scores([query]),
             "tfidf": self.fs.tfidf.scores([query]), "bm25": self.fs.bm25.scores([query])}
        if self.fs.glove is not None:
            S["glove"] = self.fs.glove.scores([query])
            S["hybrid_rrf"] = rrf([S["bm25_stem"], S["glove"]])
        t["1er_etage_ms"] = (time.perf_counter() - t0) * 1000
        scores_out = None
        if mode == "BM25":
            idx, sc = top_k_from_scores(S["bm25_stem"], top)
            ids, scores_out = idx[0], sc[0]
        elif mode == "Hybride (RRF)":
            idx, sc = top_k_from_scores(S["hybrid_rrf"], top)
            ids, scores_out = idx[0], sc[0]
        else:
            t1 = time.perf_counter()
            cand, feats = candidates_and_features([query], S, "hybrid_rrf", self.stats, min(K or self.K, len(self.texts)))
            s = self.model.predict(feats[0])
            o = np.argsort(-s)[:top]
            ids, scores_out = cand[0][o], s[o]
            t["reranking_ms"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        ans, bp = select_answer_sentence(query, [self.texts[i] for i in ids[:3]], self.stats.idf)
        t["lecture_ms"] = (time.perf_counter() - t2) * 1000
        t["total_ms"] = sum(t.values())
        passages = [{"id": int(i), "score": float(s), "text": self.texts[i]} for i, s in zip(ids, scores_out)]
        return SearchResult(passages, ans, t)

    def explain(self, query: str, passage_id: int, K: int | None = None) -> list[tuple[str, float, float]]:
        """Contributions SHAP locales (LightGBM pred_contrib) du reranker pour un passage candidat.
        Retourne [(variable, valeur, contribution)] triées par |contribution| décroissante, ou [] si le passage n'est pas candidat."""
        from .features import FEATURE_NAMES
        if self.model is None or self.fs.glove is None:
            return []
        S = {"bm25_stem": self.fs.bm25_stem.scores([query]), "lsa": self.fs.lsa.scores([query]),
             "tfidf": self.fs.tfidf.scores([query]), "glove": self.fs.glove.scores([query])}
        S["hybrid_rrf"] = rrf([S["bm25_stem"], S["glove"]])
        cand, feats = candidates_and_features([query], S, "hybrid_rrf", self.stats, min(K or self.K, len(self.texts)))
        hit = np.where(cand[0] == passage_id)[0]
        if len(hit) == 0:
            return []
        row = feats[0][hit[0]]
        contrib = self.model.predict(row.reshape(1, -1), pred_contrib=True)[0][:-1]
        out = [(n, float(v), float(c)) for n, v, c in zip(FEATURE_NAMES, row, contrib)]
        return sorted(out, key=lambda t: -abs(t[2]))
