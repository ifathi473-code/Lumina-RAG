"""Exporte le reranker LightGBM entraîné (granularité paragraphe) pour l'application."""
import json
import pickle

from common import ROOT, RES

C = pickle.load(open(RES / "cache" / "rerank_para.pkl", "rb"))
(ROOT / "models").mkdir(exist_ok=True)
C["models"]["lgbm"][1].booster_.save_model(str(ROOT / "models" / "lgbm_reranker.txt"))
(ROOT / "models" / "lgbm_reranker.json").write_text(json.dumps(
    {"features": C["feature_names"], "params": {k: (v if isinstance(v, (int, float, str)) else float(v)) for k, v in C["models"]["lgbm"][2].items()}}, indent=2))
print("modèle exporté")
