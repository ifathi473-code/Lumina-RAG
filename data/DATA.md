# Données : sources, description et préparation

Toutes les données sont **ouvertes** (consigne : privilégier l'open data). Aucune donnée personnelle n'est utilisée.

## 1. Sources

| Jeu | Rôle dans le projet | Source | Licence | SHA-256 |
|---|---|---|---|---|
| **SQuAD v1.1, jeu *dev*** | corpus de passages, questions, réponses annotées (texte + position) | Rajpurkar, Zhang, Lopyrev, Liang, *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, EMNLP 2016. Fichier : <https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json> | CC BY-SA 4.0 | `95aa6a52…966972c9` |
| **GloVe 6B, 100 d** (Wikipédia 2014 + Gigaword 5, 400 000 mots) | vecteurs de mots du récupérateur GloVe-SIF | Pennington, Socher, Manning, EMNLP 2014. Distribution *gensim-data* : <https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz> | PDDL | `4934d470…08559b43` |

Obtention : `python scripts/download_data.py` (téléchargement avec nouvelle tentative, puis vérification des empreintes ; les valeurs complètes sont dans `data/checksums.sha256`).
Les deux fichiers volumineux ne sont pas versionnés (`.gitignore`) ; les fichiers dérivés ci-dessous le sont.

## 2. Contenu et statistiques

| Élément | Valeur |
|---|---|
| Articles Wikipédia (anglais) | 48 |
| Paragraphes (passages d'origine) | 2 067 |
| Questions | 10 570 (1 à 3 réponses annotées chacune, avec position en caractères) |
| Longueur moyenne d'un paragraphe | 123 mots |
| Longueur moyenne d'une question | 10,2 mots |
| Recouvrement lexical question / paragraphe | 74,8 % des termes de la question sont dans le paragraphe |

Schéma d'une entrée : `article.title` → `paragraphs[].context` → `qas[].{id, question, answers[].{text, answer_start}}`.

## 3. Découpage (avant tout prétraitement)

| Jeu | Articles | Paragraphes | Questions |
|---|---|---|---|
| train | 24 | 1 067 | 5 238 |
| val | 8 | voir `splits.json` | 2 271 |
| test | 16 | voir `splits.json` | 3 061 |

Le découpage est fait **par article** avec la graine 42 (`ragx.data.load_squad`), pour que deux questions d'un même article ne se retrouvent jamais dans deux jeux. La liste exacte des articles de chaque jeu est dans **`data/splits.json`** ; `tests/test_no_leakage.py` vérifie l'absence de recouvrement.

## 4. Préparation (code : `src/ragx/data.py`, `src/ragx/text.py`)

1. **Lecture** du JSON → paragraphes et questions (texte, réponses, positions).
2. **Vérité terrain** : un passage est pertinent s'il appartient au paragraphe source **et** contient entièrement, par positions, une des réponses annotées. Cette définition vaut pour des passages de tailles différentes sans annotation supplémentaire.
3. **Découpage en passages** (chunking) : paragraphe entier (2 067), fenêtres de 100 mots de pas 50 (4 047), fenêtres de 50 mots de pas 25 (9 104).
4. **Prétraitement du texte** : minuscules, tokens alphanumériques, retrait des stop-words (liste scikit-learn), racinisation de Porter (NLTK), selon la configuration comparée.
5. **Statistiques non supervisées** (IDF, SVD du LSA, première composante SIF) : calculées sur le corpus de passages, jamais à partir des étiquettes de pertinence.
6. **Aucune donnée modifiée** : les fichiers sources restent intacts ; tout traitement est recalculé par les scripts.

## 5. Fichiers dérivés versionnés

| Fichier | Contenu | Script |
|---|---|---|
| `data/splits.json` | articles et effectifs de train / val / test | `scripts/13_export_splits.py` |
| `data/sample_questions.csv` | 60 questions de test (échantillon pour essayer l'application) | `scripts/13_export_splits.py` |
| `data/checksums.sha256` | empreintes des fichiers sources | `scripts/13_export_splits.py` |

## 6. Limites connues

SQuAD est en anglais et ses questions ont été écrites par des annotateurs qui avaient le paragraphe sous les yeux, d'où un recouvrement lexical élevé qui favorise BM25. Une question dont la réponse figure aussi dans un autre paragraphe peut être comptée comme un échec (faux négatif d'annotation, environ 11 % des erreurs du reranker). Les conclusions sont à lire avec cela en tête (chapitre 6 du rapport).

## 7. Utiliser vos propres documents

L'application Streamlit propose un mode « mes propres textes » (collez des paragraphes séparés par une ligne vide). Pour un corpus complet, adaptez `ragx.data.make_passages` : il suffit de fournir une liste de textes.
