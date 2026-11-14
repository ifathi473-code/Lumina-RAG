import numpy as np
import pytest

from ragx.bm25 import BM25, top_k_from_scores
from ragx.data import Passage, Question, make_passages, relevant_set
from ragx.features import FEATURE_NAMES, PassageStats, pair_features, qtype
from ragx.fusion import rank_matrix, rrf, weighted_sum
from ragx.generation import exact_match, f1_score, parse_judge
from ragx.metrics import bootstrap_ci, per_query_metrics
from ragx.text import normalize_answer, tokenize

DOCS = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Amazon rainforest covers much of northwestern Brazil.",
    "Paris is the capital of France and its largest city.",
]


def test_tokenize_stop_and_stem():
    assert tokenize("The cats are running") == ["cats", "running"]
    assert tokenize("The cats are running", stem=True) == ["cat", "run"]


def test_normalize_answer():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"


def test_bm25_ranks_relevant_first():
    b = BM25(stem=True).fit(DOCS)
    idx, sc = b.topk(["when was the Eiffel Tower completed", "how do plants use light"], 2)
    assert idx[0, 0] == 0 and idx[1, 0] == 1
    assert (np.diff(sc, axis=1) <= 1e-9).all()      # scores décroissants


def test_bm25_idf_downweights_common_terms():
    b = BM25().fit(["alpha beta", "alpha gamma", "alpha delta", "epsilon"])
    i_alpha = b.cv.vocabulary_["alpha"]; i_eps = b.cv.vocabulary_["epsilon"]
    assert b.idf[i_eps] > b.idf[i_alpha]


def test_rank_matrix_and_rrf():
    s1 = np.array([[3.0, 2.0, 1.0]]); s2 = np.array([[1.0, 2.0, 3.0]])
    assert rank_matrix(s1).tolist() == [[1, 2, 3]]
    r = rrf([s1, s2], k=60)
    assert r[0, 1] == pytest.approx(2 / 62)        # le doc du milieu est 2e partout
    assert r[0, 0] == pytest.approx(1 / 61 + 1 / 63)


def test_weighted_sum_bounds():
    s = weighted_sum([np.array([[1.0, 5.0, 3.0]])], [1.0])
    assert s.min() == 0 and s.max() == 1


def test_metrics_known_values():
    ranked = np.array([[5, 7, 9], [1, 2, 3]])
    rel = [{7}, {4}]
    m = per_query_metrics(ranked, rel, ks=(1, 3))
    assert m["hit@1"].tolist() == [0.0, 0.0]
    assert m["hit@3"].tolist() == [1.0, 0.0]
    assert m["mrr@10"].tolist() == [0.5, 0.0]
    assert m["ndcg@10"][0] == pytest.approx(1 / np.log2(3))


def test_bootstrap_ci_contains_mean():
    x = np.random.RandomState(0).binomial(1, 0.7, 500).astype(float)
    mean, lo, hi = bootstrap_ci(x)
    assert lo <= mean <= hi and hi - lo < 0.1


def test_chunking_and_relevance_by_offsets():
    text = " ".join(f"w{i}" for i in range(100))
    par = [{"parent": 0, "article": 0, "text": text}]
    ps = make_passages(par, 40, 20)
    assert len(ps) == 4 and ps[0].cstart == 0 and ps[-1].cend == len(text)
    ans = "w45 w46"
    q = Question("q", "?", 0, 0, [(ans, text.index(ans))])
    rel = relevant_set(q, {0: ps})
    assert rel == {1, 2}                              # fenêtres [20,60) et [40,80)
    assert all(ps[i].cstart <= text.index(ans) for i in rel)


def test_paragraph_granularity_one_passage_per_paragraph():
    par = [{"parent": i, "article": 0, "text": t} for i, t in enumerate(DOCS)]
    assert len(make_passages(par)) == len(DOCS)


def test_features_shape_and_qtype():
    st = PassageStats(DOCS)
    assert FEATURE_NAMES[-8:][0] == "qtype_what" and qtype("When was it built?") == 2
    cand = np.array([0, 3])
    base = {k: np.zeros(2) for k in ["bm25", "glove", "lsa", "tfidf", "rank_bm25", "rank_glove", "rank_lsa", "rrf"]}
    f = pair_features("When was the Eiffel Tower completed?", tokenize("When was the Eiffel Tower completed?", stem=True), cand, st, base)
    assert f.shape == (2, len(FEATURE_NAMES))
    cov = FEATURE_NAMES.index("coverage")
    assert f[0, cov] > f[1, cov]                       # le passage Eiffel couvre mieux la question


def test_generation_metrics():
    assert exact_match("the Eiffel Tower", ["Eiffel Tower"]) == 1.0
    assert f1_score("Eiffel", ["Eiffel Tower"]) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert f1_score("Berlin", ["Paris"]) == 0.0
    f, r = parse_judge('{"faithfulness": 1, "relevance": 0}')
    assert (f, r) == (1.0, 0.0)


def test_top_k_from_scores_sorted():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    idx, v = top_k_from_scores(s, 3)
    assert idx[0].tolist() == [1, 3, 2]
