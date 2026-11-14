# Partie 1/4 : Noyau RAG, tests et intégration continue

Compte GitHub associé : **ifathi473-code (propriétaire)**

Contenu : `src/ragx/` (BM25, dense, fusion, features, pipeline, métriques, service), `tests/` (21 tests anti-fuite inclus), `requirements*.txt`, `Makefile`, `run_all.sh`, `.github/` (CI + CODEOWNERS), `LICENSE`, `.gitignore`

Les 4 parties gardent l'arborescence d'origine de `lumina-qa` : elles ne se chevauchent pas, et une fois les 4 réunies on retrouve exactement le projet complet (les 21 tests et `make reproduce` ne fonctionnent qu'avec les 4 parties).

## Pousser cette partie
```bash
cd partie1-*/
git init -b main && git add -A
git config user.name "..." && git config user.email "<e-mail du compte ifathi473-code (propriétaire)>"
git commit -m "Partie 1 : Noyau RAG, tests et intégration continue"
git remote add origin https://github.com/ifathi473-code/lumina-qa.git   # même dépôt pour les 4 comptes
git pull origin main --allow-unrelated-histories --no-edit 2>/dev/null || true
git push -u origin main
```
Le dépôt doit être créé une fois par le propriétaire, et les 3 autres comptes ajoutés en collaborateurs (*Settings > Collaborators*). Pour 4 dépôts séparés, remplacer simplement l'URL `origin` par celle du dépôt de chaque compte.
