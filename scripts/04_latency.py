"""Mesure de latence par étage (requête unique, comme en production) et empreinte mémoire."""
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import psutil, os

warnings.filterwarnings("ignore")
from common import DATA, GLOVE, RES
from ragx.bm25 import top_k_from_scores
from ragx.data import build_qrels, load_squad, make_passages
from ragx.features import PassageStats
from ragx.pipeline import FirstStage, candidates_and_features, select_answer_sentence
from ragx.fusion import rrf

P, Q, T = load_squad(DATA)
passages = make_passages(P)
texts = [p.text for p in passages]
t0 = time.perf_counter()
fs = FirstStage(texts, str(GLOVE))
stats = PassageStats(texts)
t_build = time.perf_counter() - t0
rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
C = pickle.load(open(RES / "cache" / "rerank_para.pkl", "rb"))
lg = C["models"]["lgbm"][1]
qs = [q.text for q in Q if q.split == "test"][:300]


def ms(f):
    t = time.perf_counter(); r = f(); return r, (time.perf_counter() - t) * 1000


rows = []
for Kc in (10, 20, 50):
    tt = {"bm25": [], "glove": [], "lsa": [], "fusion+features": [], "lgbm": [], "reader": []}
    for q in qs:
        s_b, a = ms(lambda: fs.bm25_stem.scores([q])); tt["bm25"].append(a)
        s_g, a = ms(lambda: fs.glove.scores([q])); tt["glove"].append(a)
        s_l, a = ms(lambda: fs.lsa.scores([q])); tt["lsa"].append(a)
        S = {"bm25_stem": s_b, "glove": s_g, "lsa": s_l, "tfidf": fs.tfidf.scores([q]), "bm25": s_b}
        def feat():
            S["hybrid_rrf"] = rrf([S["bm25_stem"], S["glove"]])
            return candidates_and_features([q], S, "hybrid_rrf", stats, Kc)
        (cand, f), a = ms(feat); tt["fusion+features"].append(a)
        sc, a = ms(lambda: lg.predict(f[0])); tt["lgbm"].append(a)
        top = cand[0][np.argsort(-sc)][:3]
        _, a = ms(lambda: select_answer_sentence(q, [texts[p] for p in top], stats.idf)); tt["reader"].append(a)
    r = {"K": Kc}
    for k, v in tt.items():
        r[k + "_p50"] = float(np.percentile(v, 50))
    tot = np.array(tt["bm25"]) + np.array(tt["glove"]) + np.array(tt["lsa"]) + np.array(tt["fusion+features"]) + np.array(tt["lgbm"]) + np.array(tt["reader"])
    r["total_p50"], r["total_p95"] = float(np.percentile(tot, 50)), float(np.percentile(tot, 95))
    r["qps_1thread"] = 1000.0 / float(tot.mean())
    rows.append(r)
# BM25 seul (référence)
bm = [ms(lambda: top_k_from_scores(fs.bm25_stem.scores([q]), 5))[1] for q in qs]
df = pd.DataFrame(rows)
df["bm25_seul_p50"] = float(np.percentile(bm, 50))
df["index_build_s"] = t_build
df["rss_MB"] = rss
df.to_csv(RES / "latency.csv", index=False)
print(df.round(2).to_string())
