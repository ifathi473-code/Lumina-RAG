#!/usr/bin/env bash
# Reproduit TOUS les résultats du rapport (environ 10 minutes sur 2 vCPU).
# Usage :  bash run_all.sh      (SKIP_INSTALL=1 pour ne pas réinstaller les paquets)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
[ "${SKIP_INSTALL:-0}" = "1" ] || pip install -r requirements.txt
python scripts/download_data.py
python -m pytest -q tests
cd scripts
python 13_export_splits.py           # splits.json, échantillon de questions, empreintes
python 01_eda.py                     # analyse exploratoire  -> results/eda_*.{csv,json}, figures
python 02_run_experiments.py         # retrievers x chunking, fusion, rerankers, tests statistiques
python 07_export_model.py            # exporte le reranker pour l'application
python 03_interpretability.py        # permutation, SHAP, analyse d'erreurs
python 04_latency.py                 # latence par étage
python 10_error_stats.py             # statistiques d'erreurs
python 08_extra_and_figures.py       # grille BM25, ablation du prétraitement, figures de synthèse
python 11_theory_figures.py          # figures pédagogiques du chapitre théorique
python 09_make_tex_tables.py         # tableaux LaTeX du rapport (depuis results/*.csv)
python 15_build_db.py              # base SQLite data/lumina.db (tables de résultats lues par l'application)
cd ..
cp results/figures/*.png report/figures/
echo "Terminé. Rapport :  make report   (ou  cd report && xelatex rapport.tex  deux fois)"
