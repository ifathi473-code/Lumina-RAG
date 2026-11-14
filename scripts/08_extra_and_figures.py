"""Compléments (grille BM25 k1/b sur VAL, ablation du prétraitement) et figures de synthèse."""
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from common import ACCENT, DARK, DATA, FIG, GREEN, GREY, RES, style
from ragx.bm25 import BM25
from ragx.data import build_qrels, load_squad, make_passages
from ragx.metrics import per_query_metrics

style()
P, Q, T = load_squad(DATA)
ps = make_passages(P)
qr = build_qrels(Q, ps)
texts = [p.text for p in ps]
val = [q for q in Q if q.split == "val"]
test = [q for q in Q if q.split == "test"]


def ev(model, qs):
    idx = model.topk([q.text for q in qs], 50)[0]
    return per_query_metrics(idx, [qr[q.qid] for q in qs])


# 1) grille (k1, b) sur VAL -> meilleur couple appliqué au TEST
rows = []
for k1, b in itertools.product([0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9]):
    m = ev(BM25(k1, b, stem=True).fit(texts), val)
    rows.append({"k1": k1, "b": b, "val_mrr@10": m["mrr@10"].mean(), "val_ndcg@10": m["ndcg@10"].mean()})
g = pd.DataFrame(rows)
g.to_csv(RES / "bm25_grid.csv", index=False)
best = g.sort_values("val_mrr@10", ascending=False).iloc[0]
default = ev(BM25(1.2, 0.75, stem=True).fit(texts), test)
tuned = ev(BM25(best.k1, best.b, stem=True).fit(texts), test)
pd.DataFrame([
    {"config": "k1=1,2 ; b=0,75 (défaut)", "test_hit@1": default["hit@1"].mean(), "test_mrr@10": default["mrr@10"].mean()},
    {"config": f"k1={best.k1} ; b={best.b} (réglé sur val)", "test_hit@1": tuned["hit@1"].mean(), "test_mrr@10": tuned["mrr@10"].mean()},
]).to_csv(RES / "bm25_tuned_vs_default.csv", index=False)

piv = g.pivot(index="k1", columns="b", values="val_mrr@10")
fig, ax = plt.subplots(figsize=(4.8, 3.4))
im = ax.imshow(piv.values, cmap="Greens", aspect="auto")
ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
ax.set_xlabel("b (normalisation de longueur)"); ax.set_ylabel("k1 (saturation de la fréquence)")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=7, color=("white" if piv.values[i, j] > piv.values.mean() else DARK))
ax.set_title("BM25 : MRR@10 sur validation"); ax.spines[:].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "bm25_grid.png"); plt.close()

# 2) ablation du prétraitement (stop-words x racinisation) sur TEST
ab = []
for stop, stem in itertools.product([False, True], [False, True]):
    m = ev(BM25(stem=stem, stop=stop).fit(texts), test)
    ab.append({"stopwords": "oui" if stop else "non", "racinisation": "Porter" if stem else "non",
               "hit@1": m["hit@1"].mean(), "mrr@10": m["mrr@10"].mean(), "ndcg@10": m["ndcg@10"].mean()})
pd.DataFrame(ab).to_csv(RES / "preproc_ablation.csv", index=False)

# 3) figures de synthèse
R = pd.read_csv(RES / "retrieval_by_chunking.csv")
RR = pd.read_csv(RES / "rerank_results.csv")
names = {"bm25": "BM25", "bm25_stem": "BM25+Porter", "tfidf": "TF-IDF", "lsa": "LSA", "glove": "GloVe-SIF",
         "hybrid_rrf": "Hybride RRF", "hybrid_rrf3": "Hybride RRF×3", "hybrid_wsum": "Hybride pondéré"}
gl = {"para": "Paragraphe entier", "w100s50": "Fenêtre 100 mots", "w50s25": "Fenêtre 50 mots"}
fig, ax = plt.subplots(figsize=(9.2, 3.8))
w = 0.26
x = np.arange(len(names))
for i, (gk, lab) in enumerate(gl.items()):
    d = R[R.gran == gk].set_index("retriever").loc[list(names)]
    err = np.vstack([d["mrr@10"] - d["mrr@10_lo"], d["mrr@10_hi"] - d["mrr@10"]])
    ax.bar(x + (i - 1) * w, d["mrr@10"], w, yerr=err, color=[GREEN, GREY, ACCENT][i], label=lab, error_kw=dict(lw=0.8, capsize=1.5))
ax.set_xticks(x, [names[k] for k in names], rotation=25, ha="right"); ax.set_ylabel("MRR@10 (test)")
ax.set_ylim(0.3, 1.0); ax.legend(frameon=False, ncol=3, loc="upper left"); ax.set_title("Récupération : retrievers × découpage (IC bootstrap 95 %)")
plt.tight_layout(); plt.savefig(FIG / "retrievers_by_chunking.png"); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
for a, gk in zip(axs, ["para", "w50s25"]):
    d = RR[(RR.gran == gk) & (~RR.reranker.str.endswith("_K"))].copy()
    lab = {"aucun (hybride RRF)": "Sans reranking\n(hybride RRF)", "logreg": "Régression\nlogistique", "lgbm": "LightGBM\nLambdaRank", "oracle (borne sup.)": "Oracle\n(borne sup.)"}
    d = d.set_index("reranker").loc[list(lab)]
    xx = np.arange(len(d))
    a.bar(xx - 0.2, d["hit@1"], 0.4, color=GREEN, label="Hit@1")
    a.bar(xx + 0.2, d["mrr@10"], 0.4, color=GREY, label="MRR@10")
    for i, (h, m) in enumerate(zip(d["hit@1"], d["mrr@10"])):
        a.text(i - 0.2, h + 0.01, f"{h:.2f}", ha="center", fontsize=7); a.text(i + 0.2, m + 0.01, f"{m:.2f}", ha="center", fontsize=7)
    a.set_xticks(xx, [lab[k] for k in lab], fontsize=8); a.set_ylim(0, 1.1); a.set_title(gl[gk]); a.set_ylim(0, 1.25); a.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
plt.suptitle("Effet du reranking sur 50 candidats (test)", fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(FIG / "rerank_effect.png"); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 3.3))
for gk, c in zip(["para", "w50s25"], [GREEN, ACCENT]):
    d = RR[(RR.gran == gk) & (RR.reranker == "lgbm_K")].sort_values("K")
    d50 = RR[(RR.gran == gk) & (RR.reranker == "lgbm")]
    xs = list(d["K"]) + [50]; ys = list(d["mrr@10"]) + [float(d50["mrr@10"].iloc[0])]
    ax.plot(xs, ys, marker="o", color=c, label=gl[gk])
    b = R[(R.gran == gk) & (R.retriever == "bm25_stem")]["mrr@10"].iloc[0]
    ax.axhline(b, color=c, ls=":", lw=1)
ax.set_xlabel("K = nombre de candidats reranqués"); ax.set_ylabel("MRR@10"); ax.legend(frameon=False)
ax.set_title("Sensibilité à K (pointillés : BM25+Porter)")
plt.tight_layout(); plt.savefig(FIG / "k_sensitivity.png"); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 3.3))
for gk, c in zip(["para", "w100s50", "w50s25"], [GREEN, GREY, ACCENT]):
    a = pd.read_csv(RES / f"alpha_grid_{gk}.csv")
    ax.plot(a["alpha_bm25"], a["val_ndcg@10"], marker="o", ms=3, color=c, label=gl[gk])
ax.set_xlabel("poids α de BM25 (1−α pour GloVe)"); ax.set_ylabel("nDCG@10 (validation)"); ax.legend(frameon=False)
ax.set_title("Fusion pondérée : réglage de α sur validation")
plt.tight_layout(); plt.savefig(FIG / "alpha_grid.png"); plt.close()

# compromis qualité / latence
L = pd.read_csv(RES / "latency.csv").set_index("K")
pts = [("BM25+Porter", L["bm25_seul_p50"].iloc[0], R[(R.gran == "para") & (R.retriever == "bm25_stem")]["mrr@10"].iloc[0], GREY)]
pts.append(("Hybride pondéré", L["bm25_p50"].iloc[0] + L["glove_p50"].iloc[0], R[(R.gran == "para") & (R.retriever == "hybrid_wsum")]["mrr@10"].iloc[0], GREY))
for K in (10, 20, 50):
    rr = RR[(RR.gran == "para") & (RR.reranker == ("lgbm" if K == 50 else "lgbm_K")) & (RR.K == K)]["mrr@10"].iloc[0]
    pts.append((f"Hybride + LightGBM (K={K})", L.loc[K, "total_p50"], rr, GREEN))
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for n, l, m, c in pts:
    ax.scatter(l, m, s=50, color=c, zorder=3)
    ax.annotate(n, (l, m), textcoords="offset points", xytext=(6, 4 if "K=10" in n else -3), fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("latence médiane par requête (ms, échelle log)"); ax.set_ylabel("MRR@10 (test)")
ax.set_title("Compromis qualité / latence"); ax.set_xlim(0.1, 80); ax.set_ylim(0.805, 0.88)
plt.tight_layout(); plt.savefig(FIG / "quality_latency.png"); plt.close()
print(g.sort_values("val_mrr@10", ascending=False).head(3)); print(pd.read_csv(RES / "bm25_tuned_vs_default.csv")); print(pd.DataFrame(ab))
