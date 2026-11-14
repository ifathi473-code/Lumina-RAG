"""Génère les tableaux LaTeX du rapport DIRECTEMENT depuis results/*.csv (aucune valeur recopiée à la main)."""
import json

import pandas as pd

from common import RES, ROOT

OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
f3 = lambda x: f"{x:.3f}".replace(".", ",")
f2 = lambda x: f"{x:.2f}".replace(".", ",")
f1 = lambda x: f"{x:.1f}".replace(".", ",")
pct = lambda x: f"{100 * x:.1f}".replace(".", ",") + "\\,\\%"


def pfmt(p):
    return "$<0{,}001$" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf8")


NAMES = {"bm25": "BM25", "bm25_stem": "BM25 + Porter", "tfidf": "TF-IDF (cosinus)", "lsa": "LSA (256 d)", "glove": "GloVe-SIF (100 d)",
         "hybrid_rrf": "Hybride RRF (BM25+GloVe)", "hybrid_rrf3": "Hybride RRF (BM25+GloVe+LSA)", "hybrid_wsum": "Hybride pondéré (BM25+GloVe)"}
GL = {"para": "Paragraphe entier (2\\,067 passages)", "w100s50": "Fenêtres de 100 mots, pas 50 (4\\,047 passages)", "w50s25": "Fenêtres de 50 mots, pas 25 (9\\,104 passages)"}

# 1. retrievers x chunking ------------------------------------------------------------
R = pd.read_csv(RES / "retrieval_by_chunking.csv")
cols = ["hit@1", "hit@5", "hit@10", "hit@50", "mrr@10", "ndcg@10"]
L = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Récupérateur & Hit@1 & Hit@5 & Hit@10 & Hit@50 & MRR@10 & nDCG@10 \\\\"]
for g, lab in GL.items():
    d = R[R.gran == g].set_index("retriever")
    L += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\textit{{{lab}}}}} \\\\"]
    best = {c: d[c].max() for c in cols}
    for k, nm in NAMES.items():
        r = d.loc[k]
        cells = [("\\textbf{" + f3(r[c]) + "}") if abs(r[c] - best[c]) < 1e-12 else f3(r[c]) for c in cols]
        L.append(f"{nm} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_retrieval.tex", L)

# 2. reranking ----------------------------------------------------------------------------
RR = pd.read_csv(RES / "rerank_results.csv")
lab = {"aucun (hybride RRF)": "Aucun (hybride RRF, 1er étage)", "logreg": "Régression logistique", "lgbm": "LightGBM LambdaRank", "oracle (borne sup.)": "Oracle (borne supérieure)"}
L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Reranker (K = 50) & Hit@1 & Hit@5 & Hit@10 & MRR@10 [IC 95\\,\\%] & nDCG@10 \\\\"]
for g in ("para", "w50s25"):
    L += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\textit{{{GL[g]}}}}} \\\\"]
    d = RR[(RR.gran == g) & (RR.K == 50)].set_index("reranker")
    for k, nm in lab.items():
        r = d.loc[k]
        L.append(f"{nm} & {f3(r['hit@1'])} & {f3(r['hit@5'])} & {f3(r['hit@10'])} & {f3(r['mrr@10'])} [{f3(r['mrr@10_lo'])} ; {f3(r['mrr@10_hi'])}] & {f3(r['ndcg@10'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_rerank.tex", L)

# 3. sensibilité à K -------------------------------------------------------------------------
L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "K (candidats) & 5 & 10 & 20 & 30 & 50 \\\\", "\\midrule"]
for g in ("para", "w50s25"):
    d = RR[(RR.gran == g) & (RR.reranker == "lgbm_K")].set_index("K")
    d50 = RR[(RR.gran == g) & (RR.reranker == "lgbm")].iloc[0]
    vals = [d.loc[k, "mrr@10"] for k in (5, 10, 20, 30)] + [d50["mrr@10"]]
    L.append(f"MRR@10, {'paragraphe' if g == 'para' else 'fenêtre 50 mots'} & " + " & ".join(f3(v) for v in vals) + " \\\\")
    vals = [d.loc[k, "hit@1"] for k in (5, 10, 20, 30)] + [d50["hit@1"]]
    L.append(f"Hit@1, {'paragraphe' if g == 'para' else 'fenêtre 50 mots'} & " + " & ".join(f3(v) for v in vals) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_K.tex", L)

# 4. tests statistiques (paragraphe) ----------------------------------------------------------
T = pd.read_csv(RES / "stat_tests.csv")
nm2 = {"hybrid_rrf": "Hybride RRF", "bm25_stem": "BM25+Porter", "glove": "GloVe-SIF", "bm25": "BM25", "hybrid_rrf3": "Hybride RRF×3",
       "rerank_lgbm": "Hybride+LightGBM", "rerank_logreg": "Hybride+Logistique", "hybrid_rrf+none": "Hybride RRF (sans reranking)"}
L = ["\\begin{tabular}{llrrrr}", "\\toprule", "A contre B & Métrique & $\\Delta$ (A$-$B) & IC 95\\,\\% (articles) & $p$ bootstrap & $p$ Wilcoxon \\\\", "\\midrule"]
for g in ("para", "w50s25"):
    L.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{GL[g]}}}}} \\\\")
    d = T[(T.gran == g) & (T.metric.isin(["hit@1", "mrr@10"]))]
    for _, r in d.iterrows():
        if (r.A, r.B) in [("rerank_lgbm", "bm25_stem"), ("rerank_lgbm", "hybrid_rrf+none"), ("rerank_lgbm", "rerank_logreg"), ("hybrid_rrf", "bm25_stem"), ("bm25_stem", "bm25")]:
            L.append(f"{nm2[r.A]} / {nm2[r.B]} & {'Hit@1' if r.metric == 'hit@1' else 'MRR@10'} & {f3(r['diff'])} & [{f3(r.ci_lo)} ; {f3(r.ci_hi)}] & {pfmt(r.p_cluster_boot)} & {pfmt(r.p_wilcoxon)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_tests.tex", L)

# 5. proxy de lecture -------------------------------------------------------------------------
A = pd.read_csv(RES / "answer_proxy.csv")
L = ["\\begin{tabular}{lrrr}", "\\toprule", "Contexte fourni & Réponse dans le top-1 & dans le top-3 & dans le top-5 \\\\", "\\midrule"]
for s in ["BM25", "Hybride RRF", "Hybride + LightGBM"]:
    d = A[A.systeme == s].set_index("n_contexte")
    L.append(f"{s} & {pct(d.loc[1, 'reponse_dans_contexte'])} & {pct(d.loc[3, 'reponse_dans_contexte'])} & {pct(d.loc[5, 'reponse_dans_contexte'])} \\\\")
o = A[A.systeme.str.startswith("Oracle")].iloc[0]
L.append(f"Oracle (bon paragraphe) & {pct(o['reponse_dans_contexte'])} & -- & -- \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_proxy_ctx.tex", L)
L = ["\\begin{tabular}{lr}", "\\toprule", "Contexte (top-3) & Réponse dans la phrase retenue \\\\", "\\midrule"]
for s in ["BM25", "Hybride RRF", "Hybride + LightGBM"]:
    d = A[(A.systeme == s) & (A.n_contexte == 3)].iloc[0]
    L.append(f"{s} & {pct(d['reponse_dans_phrase_selectionnee'])} \\\\")
L.append(f"Oracle (bon paragraphe) & {pct(o['reponse_dans_phrase_selectionnee'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_proxy_sent.tex", L)

# 6. latence ----------------------------------------------------------------------------------
D = pd.read_csv(RES / "latency.csv")
L = ["\\begin{tabular}{rrrrrrrrr}", "\\toprule", "K & BM25 & GloVe & LSA & Fusion + caract. & LightGBM & Lecteur & Total p50 & Total p95 \\\\", "\\midrule"]
for _, r in D.iterrows():
    L.append(f"{int(r.K)} & {f2(r.bm25_p50)} & {f2(r.glove_p50)} & {f2(r.lsa_p50)} & {f2(r['fusion+features_p50'])} & {f2(r.lgbm_p50)} & {f2(r.reader_p50)} & {f1(r.total_p50)} & {f1(r.total_p95)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_latency.tex", L)

# 7. EDA --------------------------------------------------------------------------------------
E = json.loads((RES / "eda_stats.json").read_text())
S = pd.read_csv(RES / "eda_splits.csv")
L = ["\\begin{tabular}{lrrr}", "\\toprule", "Jeu & Articles & Paragraphes & Questions \\\\", "\\midrule"]
for _, r in S.iterrows():
    L.append(f"{r.split} & {r.articles} & {r.paragraphes} & {r.questions} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_splits.tex", L)
L = ["\\begin{tabular}{lrrr}", "\\toprule", "Grandeur & Moyenne & Médiane & Max \\\\", "\\midrule",
     f"Mots par paragraphe & {f1(E['mots_par_paragraphe']['moy'])} & {int(E['mots_par_paragraphe']['med'])} & {E['mots_par_paragraphe']['max']} \\\\",
     f"Mots par question & {f1(E['mots_par_question']['moy'])} & {int(E['mots_par_question']['med'])} & {E['mots_par_question']['max']} \\\\",
     f"Mots par réponse & {f1(E['mots_par_reponse']['moy'])} & {int(E['mots_par_reponse']['med'])} & {E['mots_par_reponse']['max']} \\\\",
     f"Recouvrement lexical question/paragraphe & {pct(E['recouvrement_question_paragraphe']['moy'])} & {pct(E['recouvrement_question_paragraphe']['med'])} & -- \\\\",
     "\\bottomrule", "\\end{tabular}"]
write("tab_eda.tex", L)

# 8. interprétabilité -------------------------------------------------------------------------
G = pd.read_csv(RES / "perm_importance_groups.csv")
L = ["\\begin{tabular}{lr}", "\\toprule", "Groupe de variables permuté & Baisse de MRR@10 \\\\", "\\midrule"]
for _, r in G.iterrows():
    L.append(f"{r.groupe} & {f3(r.delta_mrr)} $\\pm$ {f3(r['std'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_perm_groups.tex", L)
PM = pd.read_csv(RES / "perm_importance.csv").head(6)
SH = pd.read_csv(RES / "shap_importance.csv").head(6)
L = ["\\begin{tabular}{lrlr}", "\\toprule", "Permutation & $\\Delta$MRR & SHAP & $|\\phi|$ moyen \\\\", "\\midrule"]
for (_, a), (_, b) in zip(PM.iterrows(), SH.iterrows()):
    L.append(f"\\texttt{{{a.feature.replace('_', '\\_')}}} & {f3(a.delta_mrr)} & \\texttt{{{b.feature.replace('_', '\\_')}}} & {f3(b.mean_abs_shap)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_perm_shap.tex", L)
Er = pd.read_csv(RES / "error_categories.csv")
order = ["correct", "pertinent en rang 2-3", "pertinent en rang >3", "pertinent absent du top-50"]
lab2 = {"correct": "Correct (rang 1)", "pertinent en rang 2-3": "Pertinent classé 2\\textsuperscript{e}--3\\textsuperscript{e}", "pertinent en rang >3": "Pertinent classé au-delà du 3\\textsuperscript{e}", "pertinent absent du top-50": "Pertinent absent des 50 candidats"}
L = ["\\begin{tabular}{lr}", "\\toprule", "Issue (reranker LightGBM, test) & Part des questions \\\\", "\\midrule"]
for k in order:
    L.append(f"{lab2[k]} & {pct(Er.set_index('categorie').loc[k, 'part'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_errors.tex", L)
Eo = pd.read_csv(RES / "error_by_overlap.csv")
L = ["\\begin{tabular}{lrr}", "\\toprule", "Recouvrement question/paragraphe & Effectif & Hit@1 \\\\", "\\midrule"]
for _, r in Eo.iterrows():
    L.append(f"{r.couverture_lexicale.replace('>=', '$\\geq$').replace('<', '$<$').replace('%', '\\,\\%')} & {r.effectif} & {f3(r['hit@1'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_overlap.tex", L)

# 9. BM25 + prétraitement -----------------------------------------------------------------------
B = pd.read_csv(RES / "bm25_tuned_vs_default.csv")
L = ["\\begin{tabular}{lrr}", "\\toprule", "Configuration BM25+Porter & Hit@1 (test) & MRR@10 (test) \\\\", "\\midrule"]
for _, r in B.iterrows():
    L.append(f"{r.config.replace('.', ',')} & {f3(r['test_hit@1'])} & {f3(r['test_mrr@10'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_bm25.tex", L)
Pa = pd.read_csv(RES / "preproc_ablation.csv")
L = ["\\begin{tabular}{llrrr}", "\\toprule", "Stop-words retirés & Racinisation & Hit@1 & MRR@10 & nDCG@10 \\\\", "\\midrule"]
for _, r in Pa.iterrows():
    L.append(f"{r.stopwords} & {r.racinisation} & {f3(r['hit@1'])} & {f3(r['mrr@10'])} & {f3(r['ndcg@10'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
write("tab_preproc.tex", L)
print("tableaux:", sorted(p.name for p in OUT.glob("*.tex")))
