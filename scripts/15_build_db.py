"""Construit la base SQLite data/lumina.db : résultats d'expériences, questions d'exemple, découpage, historique.
Toutes les tables de résultats sont lues depuis results/*.csv (produits par les scripts 01 à 10) ;
la table `meta` indique pour chaque table le fichier source, le script producteur et la description.
Usage : python scripts/15_build_db.py"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "lumina.db"

PRODUCER = {
    "retrieval_by_chunking": ("02_run_experiments.py", "Retrievers (BM25, TF-IDF, LSA, GloVe, hybrides) par granularité de passage, avec IC bootstrap"),
    "rerank_results": ("02_run_experiments.py", "Rerankers (logistique, LightGBM), sensibilité à K, oracle, par granularité"),
    "stat_tests": ("02_run_experiments.py", "Tests statistiques appariés entre systèmes"),
    "alpha_grid_para": ("02_run_experiments.py", "Réglage du poids α de la fusion pondérée (paragraphes)"),
    "alpha_grid_w100s50": ("02_run_experiments.py", "Réglage de α (fenêtres de 100 mots)"),
    "alpha_grid_w50s25": ("02_run_experiments.py", "Réglage de α (fenêtres de 50 mots)"),
    "answer_proxy": ("02_run_experiments.py", "Proxy de lecture : la phrase retenue contient-elle la réponse ?"),
    "bm25_grid": ("08_extra_and_figures.py", "Grille des paramètres k1 et b de BM25"),
    "bm25_tuned_vs_default": ("08_extra_and_figures.py", "BM25 réglé contre paramètres par défaut"),
    "preproc_ablation": ("08_extra_and_figures.py", "Ablation du prétraitement (racinisation, mots vides)"),
    "eda_splits": ("01_eda.py", "Taille des jeux train / validation / test"),
    "perm_importance": ("03_interpretability.py", "Importance par permutation, variable par variable"),
    "perm_importance_groups": ("03_interpretability.py", "Importance par permutation, par groupe de variables"),
    "shap_importance": ("03_interpretability.py", "Importance SHAP moyenne par variable"),
    "shap_local_example": ("03_interpretability.py", "Explication SHAP locale d'un exemple"),
    "error_by_overlap": ("03_interpretability.py", "Hit@1 selon le recouvrement lexical question-passage"),
    "error_by_qtype": ("03_interpretability.py", "Hit@1 par type de question"),
    "error_categories": ("03_interpretability.py", "Répartition des erreurs du meilleur système"),
    "error_examples": ("03_interpretability.py", "Exemples d'erreurs annotés"),
    "latency": ("04_latency.py", "Latence par étage (p50, p95), débit et mémoire selon K"),
}


def main():
    DB.parent.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    meta = []
    for f in sorted((ROOT / "results").glob("*.csv")):
        df = pd.read_csv(f)
        df.to_sql(f.stem, con, index=False)
        prod, desc = PRODUCER.get(f.stem, ("", ""))
        meta.append((f.stem, f"results/{f.name}", prod, desc, len(df)))
    s = ROOT / "data" / "sample_questions.csv"
    if s.exists():
        df = pd.read_csv(s)
        df.to_sql("sample_questions", con, index=False)
        meta.append(("sample_questions", "data/sample_questions.csv", "13_export_splits.py", "Questions d'exemple SQuAD avec réponse annotée", len(df)))
    sp = ROOT / "data" / "splits.json"
    if sp.exists():
        j = json.loads(sp.read_text(encoding="utf8"))
        rows = [(art, split) for split, d in j["splits"].items() for art in d["articles"]]
        if rows:
            pd.DataFrame(rows, columns=["article", "split"]).to_sql("splits", con, index=False)
            meta.append(("splits", "data/splits.json", "13_export_splits.py", "Affectation de chaque article au jeu train / val / test (graine 42)", len(rows)))
    con.execute("CREATE TABLE searches (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP, question TEXT, mode TEXT, k INTEGER, top_passage INTEGER, total_ms REAL)")
    meta.append(("searches", "(écrite par l'application)", "app/streamlit_app.py", "Historique des recherches effectuées dans l'interface", 0))
    pd.DataFrame(meta, columns=["table_name", "source_file", "script", "description", "rows"]).to_sql("meta", con, index=False)
    con.commit()
    con.close()
    print("OK", DB, len(meta), "tables")


if __name__ == "__main__":
    main()
