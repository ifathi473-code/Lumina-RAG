"""Exporte les fichiers de données dérivés, versionnés dans Git :
 - data/splits.json            : articles de chaque jeu (train/val/test) et effectifs
 - data/sample_questions.csv   : 60 questions de test (échantillon pour tester l'application sans tout recalculer)
 - data/checksums.sha256       : empreintes des fichiers sources."""
import csv
import json
from collections import Counter
from common import DATA, GLOVE, ROOT
from download_data import sha256
from ragx.data import SPLIT_SEED, load_squad

P, Q, titles = load_squad(DATA)
art_split = {q.article: q.split for q in Q}
out = {"seed": SPLIT_SEED, "method": "découpage par article (aucun article partagé entre jeux)", "splits": {}}
for sp in ("train", "val", "test"):
    arts = sorted(a for a, s in art_split.items() if s == sp)
    out["splits"][sp] = {"n_articles": len(arts), "n_questions": sum(1 for q in Q if q.split == sp),
                         "n_paragraphs": sum(1 for p in P if art_split.get(p["article"]) == sp),
                         "articles": [titles[a] for a in arts]}
(ROOT / "data" / "splits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
test = [q for q in Q if q.split == "test"][::50][:60]
with open(ROOT / "data" / "sample_questions.csv", "w", newline="", encoding="utf8") as f:
    w = csv.writer(f); w.writerow(["qid", "article", "question", "reponse_annotee"])
    for q in test:
        w.writerow([q.qid, titles[q.article], q.text, q.answers[0][0] if q.answers else ""])
(ROOT / "data" / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in (DATA, GLOVE)))
print(Counter(art_split.values()), "OK : data/splits.json, sample_questions.csv, checksums.sha256")
