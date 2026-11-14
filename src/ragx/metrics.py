"""Métriques de récupération (par requête) + intervalles de confiance et tests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def per_query_metrics(ranked: np.ndarray, rel_sets: list[set[int]], ks=(1, 3, 5, 10, 20, 50)) -> dict[str, np.ndarray]:
    """ranked : (nq, K) identifiants de passages classés. Retourne des vecteurs par requête."""
    nq = ranked.shape[0]
    out: dict[str, list[float]] = {f"hit@{k}": [] for k in ks}
    out.update({f"recall@{k}": [] for k in ks})
    out["mrr@10"], out["ndcg@10"], out["p@5"] = [], [], []
    for i in range(nq):
        rel = rel_sets[i]
        row = ranked[i]
        flags = np.fromiter((int(p in rel) for p in row), dtype=np.int8, count=len(row))
        for k in ks:
            out[f"hit@{k}"].append(float(flags[:k].any()))
            out[f"recall@{k}"].append(flags[:k].sum() / max(len(rel), 1))
        top = flags[:10]
        nz = np.flatnonzero(top)
        out["mrr@10"].append(1.0 / (nz[0] + 1) if len(nz) else 0.0)
        disc = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float((top * disc).sum())
        ideal = min(len(rel), 10)
        idcg = float(disc[:ideal].sum())
        out["ndcg@10"].append(dcg / idcg if idcg > 0 else 0.0)
        out["p@5"].append(flags[:5].sum() / 5)
    return {k: np.asarray(v) for k, v in out.items()}


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cluster_bootstrap_diff(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Bootstrap par articles (clusters) de la différence moyenne a-b : plus prudent car les
    questions d'un même article sont corrélées. Retourne (diff, ic_bas, ic_haut, p_bilatéral)."""
    rng = np.random.RandomState(seed)
    uc = np.unique(clusters)
    d = a - b
    sums = np.array([d[clusters == c].sum() for c in uc])
    cnt = np.array([(clusters == c).sum() for c in uc])
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.randint(0, len(uc), len(uc))
        diffs[i] = sums[s].sum() / cnt[s].sum()
    obs = d.mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(max(p, 1.0 / n_boot), 1.0))


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
