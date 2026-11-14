"""Expériences principales : (1) retrievers x chunking, (2) fusion, (3) rerankers appris,
(4) sensibilité à K, (5) proxy de lecture extractive, (6) tests statistiques.

Protocole anti-fuite : split par ARTICLE ; hyper-paramètres choisis sur 'val' ; reranker
entraîné sur 'train' ; toutes les valeurs rapportées sont calculées sur 'test'."""
import json
import pickle
import warnings
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
from common import DATA, GLOVE, RES
from ragx.bm25 import BM25, top_k_from_scores
from ragx.data import build_qrels, load_squad, make_passages
from ragx.features import FEATURE_NAMES, PassageStats
from ragx.fusion import minmax_rows, rrf, weighted_sum
from ragx.metrics import bootstrap_ci, cluster_bootstrap_diff, per_query_metrics, wilcoxon_p
from ragx.pipeline import FirstStage, candidates_and_features, select_answer_sentence
from ragx.text import tokenize

K = 50
BATCH = 600
(RES / "cache").mkdir(exist_ok=True)
P, Q, T = load_squad(DATA)
SPL = {s: [q for q in Q if q.split == s] for s in ("train", "val", "test")}
MAIN = ["bm25", "bm25_stem", "tfidf", "lsa", "glove", "hybrid_rrf", "hybrid_rrf3"]
KEYS = ("hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "hit@50", "recall@10", "mrr@10", "ndcg@10")


def run(fs, stats, qs, qrels, feats=False, alphas=None):
    """Passe sur un ensemble de questions par lots. Retourne classements top-K par retriever."""
    out = {k: [] for k in MAIN}
    out["cand"], out["feats"] = [], []
    if alphas:
        for a in alphas:
            out[f"wsum_{a:.1f}"] = []
    for i in range(0, len(qs), BATCH):
        texts = [q.text for q in qs[i:i + BATCH]]
        S = fs.all_scores(texts)
        for k in MAIN:
            out[k].append(top_k_from_scores(S[k], K)[0])
        for a in alphas or []:
            out[f"wsum_{a:.1f}"].append(top_k_from_scores(weighted_sum([S["bm25_stem"], S["glove"]], [a, 1 - a]), K)[0])
        if feats:
            c, f = candidates_and_features(texts, S, "hybrid_rrf", stats, K)
            out["cand"].append(c)
            out["feats"].extend(f)
    for k in list(out):
        if k == "feats":
            out[k] = np.stack(out[k]) if out[k] else None
        elif out[k]:
            out[k] = np.vstack(out[k])
    return out


def metrics_of(ranked, qs, qrels):
    return per_query_metrics(ranked, [qrels[q.qid] for q in qs])


def summarize(m, extra):
    row = dict(extra)
    for k in KEYS:
        mean, lo, hi = bootstrap_ci(m[k])
        row[k] = mean
        if k in ("hit@1", "mrr@10", "ndcg@10"):
            row[k + "_lo"], row[k + "_hi"] = lo, hi
    return row


def fit_rerankers(Xtr, ytr, gtr, Xva, yva, gva):
    flat = lambda X: X.reshape(-1, X.shape[-1])
    models = {}
    # --- logistic (pointwise, standardisée), C choisi sur val (nDCG@10)
    sc = StandardScaler().fit(flat(Xtr))
    best = (-1, None)
    for C in (0.01, 0.1, 1, 10):
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(flat(Xtr)), ytr.ravel())
        s = lr.decision_function(sc.transform(flat(Xva))).reshape(Xva.shape[:2])
        v = per_query_metrics(np.argsort(-s, 1), [set(np.flatnonzero(r)) for r in yva])["ndcg@10"].mean()
        if v > best[0]:
            best = (v, (C, lr))
    models["logreg"] = (sc, best[1][1], {"C": best[1][0], "val_ndcg10": best[0]})
    # --- LightGBM LambdaRank, num_leaves choisi sur val
    best = (-1, None)
    for nl in (7, 15, 31):
        rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=800, learning_rate=0.05, num_leaves=nl,
                            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            reg_lambda=1.0, random_state=0, verbose=-1)
        rk.fit(flat(Xtr), ytr.ravel(), group=[Xtr.shape[1]] * Xtr.shape[0],
               eval_set=[(flat(Xva), yva.ravel())], eval_group=[[Xva.shape[1]] * Xva.shape[0]], eval_at=[10],
               callbacks=[lgb.early_stopping(40, verbose=False)])
        v = rk.best_score_["valid_0"]["ndcg@10"]
        if v > best[0]:
            best = (v, (nl, rk))
    models["lgbm"] = (None, best[1][1], {"num_leaves": best[1][0], "best_iter": int(best[1][1].best_iteration_), "val_ndcg10": best[0]})
    return models


def score_with(model, X):
    sc, m, _ = model
    flat = X.reshape(-1, X.shape[-1])
    if sc is not None:
        return m.decision_function(sc.transform(flat)).reshape(X.shape[:2])
    return m.predict(flat).reshape(X.shape[:2])


def labels(cand, qs, qrels):
    return np.array([[int(p in qrels[q.qid]) for p in row] for q, row in zip(qs, cand)], dtype=np.int8)


def reorder(cand, s, k=None):
    k = k or cand.shape[1]
    sub, ss = cand[:, :k], s[:, :k]
    o = np.argsort(-ss, 1, kind="stable")
    return np.take_along_axis(sub, o, 1)


def main(granularities):
    all_rows, rerank_rows, tests = [], [], []
    for gname, (size, stride) in granularities.items():
        t0 = time.time()
        passages = make_passages(P, size, stride)
        qrels = build_qrels(Q, passages)
        texts = [p.text for p in passages]
        print(f"== {gname}: {len(passages)} passages", flush=True)
        fs = FirstStage(texts, str(GLOVE))
        stats = PassageStats(texts)
        parent_of = np.array([p.parent for p in passages])

        # (a) réglage de la fusion pondérée sur VAL
        alphas = [round(a, 1) for a in np.arange(0, 1.01, 0.1)]
        rv = run(fs, stats, SPL["val"], qrels, feats=False, alphas=alphas)
        agrid = []
        for a in alphas:
            m = metrics_of(rv[f"wsum_{a:.1f}"], SPL["val"], qrels)
            agrid.append({"gran": gname, "alpha_bm25": a, "val_mrr@10": m["mrr@10"].mean(), "val_ndcg@10": m["ndcg@10"].mean()})
        agrid = pd.DataFrame(agrid)
        best_a = float(agrid.sort_values("val_ndcg@10", ascending=False).iloc[0]["alpha_bm25"])
        agrid.to_csv(RES / f"alpha_grid_{gname}.csv", index=False)
        print("  alpha*(bm25) =", best_a, flush=True)

        # (b) TEST : tous les retrievers + fusion pondérée
        need_feats = gname in RERANK_ON
        rt = run(fs, stats, SPL["test"], qrels, feats=need_feats, alphas=[best_a])
        per_q = {}
        for k in MAIN + [f"wsum_{best_a:.1f}"]:
            m = metrics_of(rt[k], SPL["test"], qrels)
            name = "hybrid_wsum" if k.startswith("wsum") else k
            per_q[name] = m
            all_rows.append(summarize(m, {"gran": gname, "n_passages": len(passages), "retriever": name}))
        pd.DataFrame(all_rows).to_csv(RES / "retrieval_by_chunking.csv", index=False)

        # (c) reranking appris (granularités sélectionnées)
        if need_feats:
            rtr = run(fs, stats, SPL["train"], qrels, feats=True)
            rv2 = run(fs, stats, SPL["val"], qrels, feats=True)
            ytr, yva = labels(rtr["cand"], SPL["train"], qrels), labels(rv2["cand"], SPL["val"], qrels)
            models = fit_rerankers(rtr["feats"], ytr, None, rv2["feats"], yva, None)
            print("  modèles :", {k: v[2] for k, v in models.items()}, flush=True)
            cand, Xte = rt["cand"], rt["feats"]
            base = metrics_of(cand, SPL["test"], qrels)            # hybride, sans reranking
            per_q["hybrid_rrf+none"] = base
            rerank_rows.append(summarize(base, {"gran": gname, "reranker": "aucun (hybride RRF)", "K": K}))
            for mname, mdl in models.items():
                s = score_with(mdl, Xte)
                m = metrics_of(reorder(cand, s), SPL["test"], qrels)
                per_q[f"rerank_{mname}"] = m
                rerank_rows.append(summarize(m, {"gran": gname, "reranker": mname, "K": K}))
                for kk in (5, 10, 20, 30):
                    m2 = metrics_of(reorder(cand, s, kk), SPL["test"], qrels)
                    rerank_rows.append(summarize(m2, {"gran": gname, "reranker": mname + "_K", "K": kk}))
            # oracle : borne supérieure du reranking (tous les pertinents du top-K en tête)
            orc = np.argsort(-labels(cand, SPL["test"], qrels), 1, kind="stable")
            m = metrics_of(np.take_along_axis(cand, orc, 1), SPL["test"], qrels)
            rerank_rows.append(summarize(m, {"gran": gname, "reranker": "oracle (borne sup.)", "K": K}))
            pd.DataFrame(rerank_rows).to_csv(RES / "rerank_results.csv", index=False)
            with open(RES / "cache" / f"rerank_{gname}.pkl", "wb") as f:
                pickle.dump({"models": models, "cand_test": cand, "X_test": Xte, "qids_test": [q.qid for q in SPL["test"]],
                             "y_test": labels(cand, SPL["test"], qrels), "feature_names": FEATURE_NAMES, "gname": gname,
                             "lgbm_ranking": reorder(cand, score_with(models["lgbm"], Xte)),
                             "logreg_ranking": reorder(cand, score_with(models["logreg"], Xte))}, f)
            # (d) proxy « lecteur extractif » : la réponse est-elle dans le contexte fourni ?
            if gname == "para":
                answer_proxy(rt, models, passages, qrels, stats)

        # (e) tests statistiques appariés (sur 'test')
        clusters = np.array([q.article for q in SPL["test"]])
        pairs = [("hybrid_rrf", "bm25_stem"), ("hybrid_rrf", "glove"), ("bm25_stem", "bm25"), ("hybrid_rrf3", "hybrid_rrf")]
        if need_feats:
            pairs += [("rerank_lgbm", "hybrid_rrf+none"), ("rerank_logreg", "hybrid_rrf+none"), ("rerank_lgbm", "rerank_logreg"), ("rerank_lgbm", "bm25_stem")]
        for a, b in pairs:
            for met in ("hit@1", "mrr@10", "ndcg@10"):
                d, lo, hi, p = cluster_bootstrap_diff(per_q[a][met], per_q[b][met], clusters)
                tests.append({"gran": gname, "A": a, "B": b, "metric": met, "diff": d, "ci_lo": lo, "ci_hi": hi,
                              "p_cluster_boot": p, "p_wilcoxon": wilcoxon_p(per_q[a][met], per_q[b][met])})
        pd.DataFrame(tests).to_csv(RES / "stat_tests.csv", index=False)
        print(f"  terminé en {time.time() - t0:.0f}s", flush=True)


def answer_proxy(rt, models, passages, qrels, stats):
    """Contexte fourni au lecteur : top-n passages. Métriques : réponse présente dans le contexte,
    puis dans la phrase sélectionnée par un lecteur extractif minimal (proxy de la génération)."""
    qs = SPL["test"]
    s = score_with(models["lgbm"], rt["feats"])
    systems = {"BM25": rt["bm25_stem"], "Hybride RRF": rt["hybrid_rrf"], "Hybride + LightGBM": reorder(rt["cand"], s)}
    gold = {q.qid: [a.lower() for a, _ in q.answers] for q in qs}
    idf = stats.idf
    rows = []
    for name, rk in systems.items():
        for n in (1, 3, 5):
            inctx, insent = [], []
            for q, row in zip(qs, rk):
                ctx = [passages[p].text for p in row[:n]]
                inctx.append(any(a in c.lower() for c in ctx for a in gold[q.qid]))
                if n == 3:
                    sent, _ = select_answer_sentence(q.text, ctx, idf)
                    insent.append(any(a in sent.lower() for a in gold[q.qid]))
            r = {"systeme": name, "n_contexte": n, "reponse_dans_contexte": float(np.mean(inctx))}
            if n == 3:
                r["reponse_dans_phrase_selectionnee"] = float(np.mean(insent))
            rows.append(r)
    # oracle : le bon paragraphe
    inctx = [any(a in passages[next(iter(qrels[q.qid]))].text.lower() for a in gold[q.qid]) for q in qs]
    sent_or = []
    for q in qs:
        ctx = [passages[next(iter(qrels[q.qid]))].text]
        sent, _ = select_answer_sentence(q.text, ctx, idf)
        sent_or.append(any(a in sent.lower() for a in gold[q.qid]))
    rows.append({"systeme": "Oracle (bon paragraphe)", "n_contexte": 1, "reponse_dans_contexte": float(np.mean(inctx)),
                 "reponse_dans_phrase_selectionnee": float(np.mean(sent_or))})
    pd.DataFrame(rows).to_csv(RES / "answer_proxy.csv", index=False)
    print(pd.DataFrame(rows), flush=True)


GRANS = {"para": (None, None), "w100s50": (100, 50), "w50s25": (50, 25)}
RERANK_ON = {"para", "w50s25"}

if __name__ == "__main__":
    only = sys.argv[1:] or list(GRANS)
    main({k: GRANS[k] for k in only})
