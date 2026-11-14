"""API REST de Lumina QA :  uvicorn app.api:app --port 8000   puis   GET /search?q=...&mode=..."""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragx.data import load_squad  # noqa: E402
from ragx.service import RagService  # noqa: E402

DATA, GLOVE = ROOT / "data" / "squad_dev-v1.1.json", ROOT / "data" / "glove-wiki-gigaword-100.gz"
if not (DATA.exists() and GLOVE.exists()):
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True)

app = FastAPI(title="Lumina QA API", version="1.0")
_P, _, _ = load_squad(ROOT / "data" / "squad_dev-v1.1.json")
svc = RagService([p["text"] for p in _P], ROOT / "data" / "glove-wiki-gigaword-100.gz", ROOT / "models" / "lgbm_reranker.txt")


@app.get("/health")
def health():
    return {"status": "ok", "passages": len(svc.texts), "modes": svc.modes}


@app.get("/search")
def search(q: str, mode: str = "Hybride + reranker LightGBM", top: int = 3, k: int = 50):
    if mode not in svc.modes:
        raise HTTPException(400, f"mode inconnu ; choisir parmi {svc.modes}")
    r = svc.search(q, mode, top, K=k)
    return {"question": q, "mode": mode, "answer_sentence": r.answer_sentence, "passages": r.passages, "timings_ms": r.timings_ms}
