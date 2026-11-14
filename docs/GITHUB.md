# Connexion à GitHub

Comptes du projet :

| Compte | Rôle |
|---|---|
| `ifathi473-code` (ifathi473@gmail.com) | propriétaire du dépôt |
| `aymenrg` (aymenrg2019@gmail.com) | collaborateur (droit d'écriture) |

## 1. Publier le dépôt (compte propriétaire)
```bash
gh auth login                                   # se connecter avec ifathi473-code
bash scripts/connect_github.sh ifathi473-code lumina-qa aymenrg
```
Le script crée le dépôt `ifathi473-code/lumina-qa`, y pousse l'historique Git et invite `aymenrg` avec le droit d'écriture. `aymenrg` accepte l'invitation (e-mail ou https://github.com/notifications), puis :
```bash
git clone https://github.com/ifathi473-code/lumina-qa.git
```

## 2. Sans GitHub CLI
```bash
git remote add origin https://github.com/ifathi473-code/lumina-qa.git
git push -u origin main
```
Puis sur GitHub : *Settings > Collaborators > Add people* > `aymenrg`.

## 3. Ajouter ensuite deux autres comptes
```bash
bash scripts/connect_github.sh ifathi473-code lumina-qa compte-3 compte-4
```
ou *Settings > Collaborators*. Ajoutez-les aussi à `.github/CODEOWNERS` pour qu'ils relisent les pull requests.

## 4. Travailler à plusieurs
Une branche par tâche (`git switch -c ma-fonction`), un push, puis une pull request relue par l'autre compte. L'intégration continue (`.github/workflows/ci.yml`) lance les 21 tests à chaque push.
Chacun configure sa propre identité : `git config user.name "..."` et `git config user.email "<e-mail du compte GitHub>"`.
