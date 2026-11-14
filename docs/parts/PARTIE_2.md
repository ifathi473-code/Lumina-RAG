# Partie 2/4 : Scripts d'expériences et d'analyse

Compte GitHub associé : **aymenrg**

Contenu : `scripts/` 01 à 06, 08, 10, 11, 13, `common.py`, `download_data.py` (EDA, retrievers, reranking, interprétabilité, latence, figures, splits)

Les 4 parties gardent l'arborescence d'origine de `lumina-qa` : elles ne se chevauchent pas, et une fois les 4 réunies on retrouve exactement le projet complet (les 21 tests et `make reproduce` ne fonctionnent qu'avec les 4 parties).

## Pousser cette partie
```bash
cd partie2-*/
git init -b main && git add -A
git config user.name "..." && git config user.email "<e-mail du compte aymenrg>"
git commit -m "Partie 2 : Scripts d'expériences et d'analyse"
git remote add origin https://github.com/ifathi473-code/lumina-qa.git   # même dépôt pour les 4 comptes
git pull origin main --allow-unrelated-histories --no-edit 2>/dev/null || true
git push -u origin main
```
Le dépôt doit être créé une fois par le propriétaire, et les 3 autres comptes ajoutés en collaborateurs (*Settings > Collaborators*). Pour 4 dépôts séparés, remplacer simplement l'URL `origin` par celle du dépôt de chaque compte.
