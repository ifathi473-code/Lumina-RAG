# Partie 3/4 : Résultats, modèles et données

Compte GitHub associé : **compte-3**

Contenu : `results/` (CSV + figures), `models/` (reranker LightGBM), `data/` (splits, échantillon, checksums, SQLite), `scripts/07_export_model.py`, `09_make_tex_tables.py`, `15_build_db.py`

Les 4 parties gardent l'arborescence d'origine de `lumina-qa` : elles ne se chevauchent pas, et une fois les 4 réunies on retrouve exactement le projet complet (les 21 tests et `make reproduce` ne fonctionnent qu'avec les 4 parties).

## Pousser cette partie
```bash
cd partie3-*/
git init -b main && git add -A
git config user.name "..." && git config user.email "<e-mail du compte compte-3>"
git commit -m "Partie 3 : Résultats, modèles et données"
git remote add origin https://github.com/ifathi473-code/lumina-qa.git   # même dépôt pour les 4 comptes
git pull origin main --allow-unrelated-histories --no-edit 2>/dev/null || true
git push -u origin main
```
Le dépôt doit être créé une fois par le propriétaire, et les 3 autres comptes ajoutés en collaborateurs (*Settings > Collaborators*). Pour 4 dépôts séparés, remplacer simplement l'URL `origin` par celle du dépôt de chaque compte.
