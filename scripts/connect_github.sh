#!/usr/bin/env bash
# Publie le dépôt sur GitHub et ajoute les collaborateurs.
# Usage :  bash scripts/connect_github.sh [propriétaire] [nom-du-dépôt] [collaborateur ...]
# Exemple : bash scripts/connect_github.sh ifathi473-code lumina-qa aymenrg
# Pour ajouter plus tard d'autres comptes : bash scripts/connect_github.sh ifathi473-code lumina-qa autre-compte-1 autre-compte-2
# Prérequis : GitHub CLI (gh) connecté au compte propriétaire :  gh auth login
set -euo pipefail
OWNER="${1:-ifathi473-code}"; REPO="${2:-lumina-qa}"; shift 2 2>/dev/null || true
COLLABS=("${@:-aymenrg}")
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo "Installez GitHub CLI : https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Connectez-vous d'abord :  gh auth login"; exit 1; }
[ -d .git ] || { git init -q -b main && git add -A && git commit -qm "Version initiale de Lumina QA"; }
if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$OWNER/$REPO.git"
  git push -u origin main
else
  gh repo create "$OWNER/$REPO" --public --source . --remote origin --push \
     --description "Lumina QA : recherche documentaire, reranking appris et réponses sourcées"
fi
for c in "${COLLABS[@]}"; do
  gh api -X PUT "repos/$OWNER/$REPO/collaborators/$c" -f permission=push >/dev/null && echo "Invitation envoyée à $c (à accepter par e-mail ou sur github.com/notifications)"
done
echo "Dépôt : https://github.com/$OWNER/$REPO"
