<p align="center"><img src="assets/lumina_logo.png" width="360" alt="Lumina QA"></p>

# Optimisation et déploiement d'un système RAG pour la question-réponse
**Étude comparative des stratégies de récupération et de reranking, et cadre d'évaluation rigoureux**

Groupe 1 : Ismail Fathi, Riyad Sammouh, Fathi Ismail, Rguieg Aymen

## 1. Contenu du dépôt
Un pipeline RAG complet et reproductible :

```
question -> 1er étage (BM25, TF-IDF, LSA, GloVe-SIF, hybrides) -> top-K candidats
         -> reranker appris (régression logistique, LightGBM LambdaRank)
         -> lecteur extractif -> réponse + passages sources
```
Chunking paramétrable, découpage par article **avant** tout prétraitement (pas de fuite), métriques Hit@k / MRR / nDCG avec IC bootstrap par articles et tests appariés, interprétabilité (permutation, SHAP, analyse d'erreurs), mesure de latence, application Streamlit, API FastAPI, Docker.

## 2. Prérequis
Python 3.10+, ≈ 1 Go d'espace disque, 2 Go de RAM. Aucun GPU. Docker est optionnel.

## 3. Installation
```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                       # SQuAD v1.1 dev + GloVe 100d, vérification SHA-256
```

## 4. Reproduire les résultats (≈ 10 min sur 2 vCPU)
```bash
make reproduce        # équivalent à : bash run_all.sh
make test             # 21 tests dont les garde-fous anti-fuite
make report           # compile le rapport (XeLaTeX)
```
| Script | Rôle |
|---|---|
| `download_data.py` | télécharge et vérifie les données |
| `01_eda.py` | analyse exploratoire et découpage |
| `02_run_experiments.py` | retrievers, chunking, fusion, rerankers, sensibilité à K, tests |
| `03_interpretability.py` | permutation, SHAP, erreurs |
| `04_latency.py` | latence, débit, mémoire |
| `05_neural_optional.py`, `06_generation_optional.py` | extensions Hugging Face (bi-encodeur, cross-encodeur, flan-T5) |
| `07_export_model.py` | exporte le reranker pour la démo |
| `08_extra_and_figures.py`, `10_error_stats.py`, `11_theory_figures.py` | grilles, statistiques d'erreurs, figures |
| `09_make_tex_tables.py` | tableaux LaTeX générés depuis `results/*.csv` |
| `12_video_script.py` | script de la vidéo de présentation (uniquement si `slides/` est présent) |
| `13_export_splits.py` | exporte `data/splits.json` (identifiants des articles par jeu) |
| `15_build_db.py` | construit la base SQLite `data/lumina.db` (22 tables de résultats, catalogue `meta`, historique `searches`) |
| `connect_github.sh` | publie le dépôt sur GitHub et invite les collaborateurs (voir `docs/GITHUB.md`) |
| `14_record_demo.py` | enregistre la vidéo de démonstration (Playwright + ffmpeg) |

Graine fixée (42). Une réexécution indépendante a redonné des CSV identiques pour `alpha_grid_*`, `answer_proxy`, `eda_splits`, `rerank_results`, `retrieval_by_chunking`, `stat_tests`.

## 5. Données
Les résultats d'expériences et l'historique de l'application sont stockés dans la base SQLite `data/lumina.db` (reconstruite par `python scripts/15_build_db.py`). SQuAD v1.1 (dev, licence CC BY-SA 4.0) et GloVe wiki-gigaword 100d. Sources, prétraitement et découpage : [`data/DATA.md`](data/DATA.md). Le découpage (24 / 8 / 16 articles) est publié dans `data/splits.json`, avec un échantillon de questions (`data/sample_questions.csv`) et les empreintes `data/checksums.sha256`.

## 6. Démo et déploiement
```bash
make app                              # Streamlit : http://localhost:8501
make api                              # API : GET /health, GET /search?q=...&mode=...
docker compose up --build             # Streamlit (8501) + API (8000)
```
**Lumina QA** compte cinq onglets : **Recherche** (réponse, sources surlignées, bon passage annoté, pipeline et latence, explication SHAP du score, export JSON/CSV, historique), **Comparer** (stratégies côte à côte, mini-évaluation en direct), **Performances** (graphiques lus dans la base, avec la source indiquée sous chacun), **Base de données** (catalogue des tables, explorateur, requêtes SQL en lecture seule, schéma du parcours des données), **Fonctionnement** (schéma et formules) et **À propos**. L'utilisateur peut aussi déposer ses propres fichiers (.txt, .md, .pdf) ou coller du texte, et régler le nombre de candidats K du reranker. Une **vidéo de démonstration** est fournie : `demo/demo_plateforme_RAG.mp4` (régénérable avec `python scripts/14_record_demo.py` quand l'app et l'API tournent). Les données manquantes sont téléchargées au premier lancement. Détails (Streamlit Community Cloud, Docker, CI, exemple d'appel API) : [`docs/DEPLOIEMENT.md`](docs/DEPLOIEMENT.md). Captures : `docs/screens/`. Intégration continue : `.github/workflows/ci.yml`.

## 7. Résultats principaux (test : 3 061 questions, 16 articles jamais vus)
| Système | Hit@1 | MRR@10 | nDCG@10 |
|---|---|---|---|
| BM25 + Porter | 0,788 | 0,848 | 0,874 |
| GloVe-SIF | 0,369 | 0,472 | 0,527 |
| Hybride RRF | 0,559 | 0,667 | 0,721 |
| Hybride pondéré | 0,787 | 0,847 | 0,873 |
| **Hybride + reranker LightGBM** | **0,816** | **0,868** | **0,890** |

Gain du reranker sur BM25 : +2,8 points de Hit@1 (IC 95 % [1,8 ; 3,9]), ≈ 8 ms par requête. Analyse complète dans le rapport (livré séparément).

## 8. Structure
```
src/ragx/   bibliothèque        scripts/  expériences numérotées
app/        Streamlit + API     tests/    unitaires et anti-fuite
results/    CSV, JSON, figures  models/   reranker exporté
demo/       vidéo de démonstration (mp4)
data/       DATA.md, splits     docs/     déploiement, captures
```

## 9. Technologies
scikit-learn, LightGBM, SHAP, NLTK, NumPy/pandas, Streamlit, FastAPI, pytest, Docker. Extensions optionnelles : PyTorch, Hugging Face (`requirements-neural.txt`).

## 10. Limites
Les modèles neuronaux Hugging Face (scripts 05 et 06) sont fournis mais n'ont pas été exécutés faute d'accès réseau ; leurs tableaux s'ajoutent au rapport s'ils existent. L'image Docker n'a pas pu être construite dans l'environnement de développement (pas de démon Docker) ; les commandes sont celles de la documentation.

## 11. Licence
Code sous licence MIT (`LICENSE`). Données : voir `data/DATA.md`.
