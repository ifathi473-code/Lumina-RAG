"""Interprétabilité du reranker : importance par permutation (ΔMRR), SHAP, analyse d'erreurs."""
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
from common import DATA, FIG, GREEN, GREY, ACCENT, DARK, RES, style
from ragx.data import build_qrels, load_squad, make_passages
from ragx.features import FEATURE_NAMES, QTYPES
from ragx.metrics import per_query_metrics

style()
GN = "para"
C = pickle.load(open(RES / "cache" / f"rerank_{GN}.pkl", "rb"))
P, Q, T = load_squad(DATA)
passages = make_passages(P)
qrels = build_qrels(Q, passages)
qs = [q for q in Q if q.split == "test"]
assert [q.qid for q in qs] == C["qids_test"]
rel = [qrels[q.qid] for q in qs]
cand, X = C["cand_test"], C["X_test"]
model = C["models"]["lgbm"][1]


def mrr(Xp):
    s = model.predict(Xp.reshape(-1, Xp.shape[-1])).reshape(Xp.shape[:2])
    o = np.argsort(-s, 1, kind="stable")
    return per_query_metrics(np.take_along_axis(cand, o, 1), rel)["mrr@10"].mean()


# 1) Importance par permutation : on permute UNE caractéristique entre les candidats d'une même
#    requête (casse le lien avec la pertinence, garde la distribution) ; 10 répétitions.
rng = np.random.RandomState(0)
base = mrr(X)
rows = []
for j, name in enumerate(FEATURE_NAMES):
    drops = []
    for _ in range(10):
        Xp = X.copy()
        perm = rng.rand(X.shape[0], X.shape[1]).argsort(1)
        Xp[:, :, j] = np.take_along_axis(X[:, :, j], perm, 1)
        drops.append(base - mrr(Xp))
    rows.append({"feature": name, "delta_mrr": np.mean(drops), "std": np.std(drops)})
perm = pd.DataFrame(rows).sort_values("delta_mrr", ascending=False)
perm.to_csv(RES / "perm_importance.csv", index=False)


# 1bis) Importance par GROUPES de variables corrélées (permutation simultanée du groupe)
GROUPS = {
    "Scores lexicaux (BM25, TF-IDF, rangs)": ["bm25", "bm25_rel", "tfidf_cos", "rank_bm25"],
    "Scores sémantiques (GloVe, LSA, rangs)": ["glove_cos", "lsa_cos", "rank_glove", "rank_lsa", "rrf"],
    "Appariement fin (couverture, bigrammes, proximité, phrase)": ["coverage", "idf_coverage", "bigram_match", "bigram_frac", "proximity", "first_match_pos", "best_sent_cov"],
    "Longueurs": ["len_passage", "len_query"],
    "Type de réponse attendu (nombres, années, majuscules, wh-)": ["n_numbers", "n_years", "n_caps"] + [f"qtype_{t}" for t in QTYPES],
}
grows = []
for gname_, feats_ in GROUPS.items():
    cols = [FEATURE_NAMES.index(f) for f in feats_]
    drops = []
    for _ in range(10):
        Xp = X.copy()
        perm_ = rng.rand(X.shape[0], X.shape[1]).argsort(1)
        for j in cols:
            Xp[:, :, j] = np.take_along_axis(X[:, :, j], perm_, 1)
        drops.append(base - mrr(Xp))
    grows.append({"groupe": gname_, "delta_mrr": np.mean(drops), "std": np.std(drops)})
gdf = pd.DataFrame(grows).sort_values("delta_mrr", ascending=False)
gdf.to_csv(RES / "perm_importance_groups.csv", index=False)
print(gdf)

top = perm.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.barh(top["feature"], top["delta_mrr"], xerr=top["std"], color=GREEN, ecolor=GREY)
ax.set_xlabel("Baisse de MRR@10 quand la variable est permutée"); ax.set_title("Importance par permutation (reranker LightGBM)")
plt.tight_layout(); plt.savefig(FIG / "perm_importance.png"); plt.close()

# 2) SHAP (TreeExplainer) sur un échantillon de paires du jeu de test
idx = rng.choice(X.shape[0], 400, replace=False)
Xs = X[idx].reshape(-1, X.shape[-1])
expl = shap.TreeExplainer(model)
sv = expl.shap_values(Xs)
sv = np.asarray(sv)
mean_abs = np.abs(sv).mean(0)
shap_df = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(RES / "shap_importance.csv", index=False)
plt.figure(figsize=(7, 4.6))
shap.summary_plot(sv, Xs, feature_names=FEATURE_NAMES, max_display=12, show=False, plot_size=None, color_bar=True)
plt.title("SHAP : contribution des variables au score du reranker", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "shap_summary.png"); plt.close()

# Explication locale d'un cas corrigé par le reranker (rang >1 avant -> rang 1 après)
before = C["cand_test"]
after = C["lgbm_ranking"]
local = None
for i, q in enumerate(qs):
    r0 = [p in rel[i] for p in before[i]]
    r1 = [p in rel[i] for p in after[i]]
    if r0[0] is False and r1[0] is True and np.argmax(r0) >= 2:
        local = i
        break
if local is not None:
    i = local
    pos = int(np.flatnonzero([p in rel[i] for p in cand[i]])[0])
    s1 = expl.shap_values(X[i][[pos]])[0]
    top_j = np.argsort(-np.abs(s1))[:6]
    loc = pd.DataFrame({"feature": [FEATURE_NAMES[j] for j in top_j], "valeur": X[i][pos][top_j], "shap": s1[top_j]})
    loc.to_csv(RES / "shap_local_example.csv", index=False)
    (RES / "shap_local_example.txt").write_text(f"Question: {qs[i].text}\nRang avant reranking: {pos + 1}\nPassage: {passages[cand[i][pos]].text[:300]}...\n")

# 3) Analyse d'erreurs (top-1 après reranking incorrect)
lg = C["lgbm_ranking"]
top1_ok = np.array([lg[i][0] in rel[i] for i in range(len(qs))])
in_cand = np.array([bool(set(cand[i]) & rel[i]) for i in range(len(qs))])
rank_gold = np.array([(int(np.flatnonzero([p in rel[i] for p in lg[i]])[0]) + 1) if in_cand[i] else 99 for i in range(len(qs))])
cat = np.where(top1_ok, "correct", np.where(~in_cand, "pertinent absent du top-50", np.where(rank_gold <= 3, "pertinent en rang 2-3", "pertinent en rang >3")))
err = pd.Series(cat).value_counts(normalize=True).rename("part").reset_index().rename(columns={"index": "categorie", "": "categorie"})
err.columns = ["categorie", "part"]
err.to_csv(RES / "error_categories.csv", index=False)

qtypes = np.array([QTYPES[int(np.argmax(X[i][0][-len(QTYPES):]))] for i in range(len(qs))])
cov = X[:, 0, FEATURE_NAMES.index("coverage")]  # indicatif : couverture du 1er candidat
qcov = []
from ragx.text import tokenize
ptok = {p.pid: set(tokenize(p.text, stem=True)) for p in passages}
for i, q in enumerate(qs):
    qt = set(tokenize(q.text, stem=True))
    g = next(iter(rel[i]))
    qcov.append(len(qt & ptok[g]) / max(len(qt), 1))
qcov = np.array(qcov)
bins = pd.cut(qcov, [-0.01, 0.4, 0.6, 0.8, 1.01], labels=["<40%", "40-60%", "60-80%", ">=80%"])
tab_q = pd.DataFrame({"type": qtypes, "ok": top1_ok}).groupby("type")["ok"].agg(["mean", "count"]).reset_index()
tab_q.columns = ["type_question", "hit@1", "effectif"]
tab_q.sort_values("hit@1").to_csv(RES / "error_by_qtype.csv", index=False)
tab_c = pd.DataFrame({"couv": bins, "ok": top1_ok}).groupby("couv", observed=True)["ok"].agg(["mean", "count"]).reset_index()
tab_c.columns = ["couverture_lexicale", "hit@1", "effectif"]
tab_c.to_csv(RES / "error_by_overlap.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(tab_c["couverture_lexicale"].astype(str), tab_c["hit@1"], color=GREEN)
for i, (v, n) in enumerate(zip(tab_c["hit@1"], tab_c["effectif"])):
    ax[0].text(i, v + 0.01, f"n={n}", ha="center", fontsize=8)
ax[0].set_ylim(0, 1.08); ax[0].set_title("Hit@1 selon le recouvrement question/paragraphe")
tq = tab_q.sort_values("hit@1")
ax[1].barh(tq["type_question"], tq["hit@1"], color=GREY); ax[1].set_xlim(0, 1.0); ax[1].set_title("Hit@1 selon le type de question")
plt.tight_layout(); plt.savefig(FIG / "error_analysis.png"); plt.close()

# exemples qualitatifs d'erreurs (3 par catégorie) -> pour le rapport
ex = []
for c in ["pertinent en rang 2-3", "pertinent en rang >3", "pertinent absent du top-50"]:
    ids = np.flatnonzero(cat == c)[:3]
    for i in ids:
        ex.append({"categorie": c, "question": qs[i].text, "reponse_attendue": qs[i].answers[0][0],
                   "top1": passages[lg[i][0]].text[:220].replace("\n", " ")})
pd.DataFrame(ex).to_csv(RES / "error_examples.csv", index=False)
print(perm.head(8)); print(shap_df.head(8)); print(err); print(tab_q); print(tab_c)
