"""Vérifie les fichiers de données dérivés versionnés (data/splits.json, sample_questions.csv)."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_splits_disjoints_et_effectifs():
    s = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf8"))
    sets = {k: set(v["articles"]) for k, v in s["splits"].items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert [s["splits"][k]["n_articles"] for k in ("train", "val", "test")] == [24, 8, 16]
    assert sum(v["n_questions"] for v in s["splits"].values()) == 10570
    assert s["seed"] == 42


def test_echantillon_questions_issu_du_test():
    s = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf8"))
    test_articles = set(s["splits"]["test"]["articles"])
    with open(ROOT / "data" / "sample_questions.csv", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60 and all(r["article"] in test_articles for r in rows)


def test_empreintes_presentes():
    lines = (ROOT / "data" / "checksums.sha256").read_text().strip().splitlines()
    assert len(lines) == 2 and all(len(l.split()[0]) == 64 for l in lines)
