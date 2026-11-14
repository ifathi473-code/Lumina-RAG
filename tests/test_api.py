"""Test de l'API FastAPI (nécessite les données et le modèle exporté)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "squad_dev-v1.1.json").exists() or not (ROOT / "models" / "lgbm_reranker.txt").exists(),
                                reason="données ou modèle absents")


def test_health_et_search():
    from fastapi.testclient import TestClient
    from app.api import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/search", params={"q": "Which NFL team represented the AFC at Super Bowl 50?", "mode": "Hybride + reranker LightGBM", "top": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["passages"] and body["answer_sentence"] and "timings_ms" in body


def test_explain_contributions():
    from ragx.service import RagService
    import pytest
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "glove-wiki-gigaword-100.gz").exists():
        pytest.skip("GloVe absent")
    texts = ["The Yuan dynasty ruled China.", "Paris is the capital of France.", "The Ming dynasty followed the Yuan dynasty in China.",
             "Bananas are rich in potassium.", "Rivers flow to the sea."] * 12
    svc = RagService(texts, root / "data" / "glove-wiki-gigaword-100.gz", root / "models" / "lgbm_reranker.txt", K=20)
    r = svc.search("What dynasty came after the Yuan?", "Hybride + reranker LightGBM", 3, K=20)
    contrib = svc.explain("What dynasty came after the Yuan?", r.passages[0]["id"], 20)
    assert contrib and all(len(t) == 3 for t in contrib)
    assert abs(contrib[0][2]) >= abs(contrib[-1][2])
