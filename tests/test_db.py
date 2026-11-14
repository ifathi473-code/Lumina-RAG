import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_db_matches_csv(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "15_build_db.py")], check=True, capture_output=True)
    con = sqlite3.connect(ROOT / "data" / "lumina.db")
    meta = pd.read_sql("SELECT * FROM meta", con)
    assert {"retrieval_by_chunking", "rerank_results", "latency", "splits", "searches"} <= set(meta.table_name)
    for t in ["retrieval_by_chunking", "rerank_results", "latency"]:
        db = pd.read_sql(f'SELECT * FROM "{t}"', con)
        csv = pd.read_csv(ROOT / "results" / f"{t}.csv")
        assert db.shape == csv.shape
    sp = pd.read_sql("SELECT split, COUNT(*) n FROM splits GROUP BY split", con).set_index("split")["n"]
    assert sp.to_dict() == {"test": 16, "train": 24, "val": 8}
    assert pd.read_sql("SELECT COUNT(DISTINCT article) n FROM splits", con)["n"][0] == 48  # aucun article dans deux jeux
