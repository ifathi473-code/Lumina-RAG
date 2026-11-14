"""EXTENSION NEURONALE (optionnelle) : bi-encodeur + cross-encodeur Hugging Face.

À lancer sur une machine avec accès à Hugging Face :
    pip install sentence-transformers
    python scripts/05_neural_optional.py --n 1500
Produit results/neural_results.csv et results/neural_table.tex (inclus automatiquement par le rapport)."""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from common import DATA, GLOVE, RES, ROOT
from ragx.bm25 import top_k_from_scores
from ragx.data import build_qrels, load_squad, make_passages
from ragx.fusion import rrf
from ragx.metrics import bootstrap_ci, per_query_metrics
from ragx.neural import BiEncoder, CrossEncoderReranker
from ragx.pipeline import FirstStage

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500, help="nb de questions de test")
ap.add_argument("--bi", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--ce", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--K", type=int, default=20)
a = ap.parse_args()

P, Q, T = load_squad(DATA)
passages = make_passages(P)
qrels = build_qrels(Q, passages)
texts = [p.text for p in passages]
qs = [q for q in Q if q.split == "test"][: a.n]
rel = [qrels[q.qid] for q in qs]
qt = [q.text for q in qs]
fs = FirstStage(texts, str(GLOVE))
bi = BiEncoder(a.bi).fit(texts)
ce = CrossEncoderReranker(a.ce)

S_bm, S_bi = fs.bm25_stem.scores(qt), bi.scores(qt)
systems = {"BM25 (stem)": S_bm, "Bi-encodeur " + a.bi.split("/")[-1]: S_bi, "Hybride RRF BM25 + bi-encodeur": rrf([S_bm, S_bi])}
rows = []
for name, S in systems.items():
    t = time.time()
    cand = top_k_from_scores(S, 50)[0]
    m = per_query_metrics(cand, rel)
    rows.append({"systeme": name, **{k: m[k].mean() for k in ("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")}, "lat_ms_par_req": 0.0})
# cross-encodeur sur le top-K de l'hybride
S = systems["Hybride RRF BM25 + bi-encodeur"]
cand = top_k_from_scores(S, a.K)[0]
t = time.time()
rr = np.zeros_like(cand)
for i, q in enumerate(qt):
    sc = ce.rerank(q, [texts[p] for p in cand[i]])
    rr[i] = cand[i][np.argsort(-sc)]
lat = (time.time() - t) / len(qt) * 1000
m = per_query_metrics(rr, rel)
rows.append({"systeme": f"Hybride + cross-encodeur (K={a.K})", **{k: m[k].mean() for k in ("hit@1", "hit@5", "hit@10", "mrr@10", "ndcg@10")}, "lat_ms_par_req": lat})
df = pd.DataFrame(rows)
df.to_csv(RES / "neural_results.csv", index=False)
tex = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Système & Hit@1 & Hit@5 & Hit@10 & MRR@10 & nDCG@10 \\\\", "\\midrule"]
for _, r in df.iterrows():
    tex.append(f"{r['systeme']} & {r['hit@1']:.3f} & {r['hit@5']:.3f} & {r['hit@10']:.3f} & {r['mrr@10']:.3f} & {r['ndcg@10']:.3f} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(RES / "neural_table.tex").write_text("\n".join(tex)); (ROOT / "report" / "tables").mkdir(parents=True, exist_ok=True); (ROOT / "report" / "tables" / "neural_table.tex").write_text("\n".join(tex))
print(df.round(3).to_string())
