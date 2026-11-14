# Déploiement

L'application a trois modes de recherche (BM25, hybride RRF, hybride + reranker LightGBM), exposés de deux façons : une interface **Streamlit** et une **API REST FastAPI**. Les deux utilisent la même classe `ragx.service.RagService`.

## 1. En local (sans Docker)
```bash
pip install -r requirements.txt
python scripts/download_data.py
export PYTHONPATH=src                      # Windows PowerShell : $env:PYTHONPATH="src"
streamlit run app/streamlit_app.py         # http://localhost:8501
uvicorn app.api:app --port 8000            # http://localhost:8000/docs
```
Le premier chargement construit l'index (environ 6 s) ; il est ensuite gardé en cache.

## 2. Docker
```bash
docker build -t rag-ecc .
docker run -p 8501:8501 rag-ecc                                              # interface
docker run -p 8000:8000 rag-ecc uvicorn app.api:app --host 0.0.0.0 --port 8000   # API
```
Le conteneur utilise `requirements-deploy.txt` (sans SHAP ni pytest), tourne avec un utilisateur non administrateur et déclare un `HEALTHCHECK`. Les données ouvertes sont téléchargées à la construction et leurs empreintes sont vérifiées.

## 3. Docker Compose (interface + API)
```bash
docker compose up --build
```
| Service | URL | Contrôle de santé |
|---|---|---|
| `app` (Streamlit) | http://localhost:8501 | `/_stcore/health` |
| `api` (FastAPI) | http://localhost:8000/docs | `/health` |

## 4. En ligne, gratuitement
* **Streamlit Community Cloud** : pousser le dépôt sur GitHub, *New app*, fichier principal `app/streamlit_app.py`, Python 3.11. Les dépendances sont lues dans `requirements.txt`. Les données sont téléchargées au premier lancement si `data/` est vide (voir `app/streamlit_app.py`).
* **Hugging Face Spaces (Docker)** : créer un Space de type Docker et y pousser le dépôt ; ajouter `app_port: 8501` dans l'en-tête du README du Space.

## 5. Utiliser l'API
```bash
curl "http://localhost:8000/health"
curl --get "http://localhost:8000/search" \
     --data-urlencode "q=Which NFL team represented the AFC at Super Bowl 50?" \
     --data-urlencode "mode=Hybride + reranker LightGBM" --data-urlencode "top=3"
```
Réponse JSON : `answer_sentence` (phrase retenue par le lecteur), `passages` (texte et score), `timings_ms` (latence par étage).

## 6. Performances observées
Environ 8 ms par requête (médiane, 1 cœur) pour le pipeline complet, soit environ 116 requêtes par seconde et par cœur ; mémoire résidente d'environ 430 Mo une fois l'index construit. Détail dans le chapitre 7 du rapport.

## 7. Dépannage
| Symptôme | Cause probable | Remède |
|---|---|---|
| `ModuleNotFoundError: ragx` | `PYTHONPATH` non défini | `export PYTHONPATH=src` |
| Téléchargement bloqué | réseau restreint | placer les deux fichiers dans `data/` à la main (voir `data/DATA.md`) |
| Premier chargement lent | construction de l'index | normal, mis en cache ensuite |
| Port occupé | autre service sur 8501 / 8000 | changer le port : `-p 8502:8501` |

## 8. Limites
Le service charge tout l'index en mémoire, ce qui convient à quelques milliers de passages. Pour un corpus de plusieurs millions de passages, il faudrait un index approché (HNSW) et un stockage externe.
