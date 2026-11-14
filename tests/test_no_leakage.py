"""Garde-fous anti-fuite : le découpage train/val/test ne partage aucun article."""
from pathlib import Path

import pytest

from ragx.data import load_squad

DATA = Path(__file__).resolve().parents[1] / "data" / "squad_dev-v1.1.json"


@pytest.mark.skipif(not DATA.exists(), reason="données non téléchargées (python scripts/download_data.py)")
def test_splits_disjoint_by_article():
    P, Q, T = load_squad(DATA)
    arts = {s: {q.article for q in Q if q.split == s} for s in ("train", "val", "test")}
    assert not (arts["train"] & arts["val"]) and not (arts["train"] & arts["test"]) and not (arts["val"] & arts["test"])
    assert sum(len(v) for v in arts.values()) == len(T)
    assert len(Q) == 10570


@pytest.mark.skipif(not DATA.exists(), reason="données non téléchargées")
def test_split_is_deterministic():
    _, Q1, _ = load_squad(DATA)
    _, Q2, _ = load_squad(DATA)
    assert [q.split for q in Q1] == [q.split for q in Q2]
