"""Analyse exploratoire : tailles, types de questions, position des réponses, recouvrement lexical."""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, FIG, RES, GREEN, GREY, ACCENT, style
from ragx.data import load_squad
from ragx.features import QTYPES, qtype
from ragx.text import tokenize

style()
P, Q, T = load_squad(DATA)
ptxt = {p["parent"]: p["text"] for p in P}

plen = np.array([len(p["text"].split()) for p in P])
qlen = np.array([len(q.text.split()) for q in Q])
alen = np.array([len(q.answers[0][0].split()) for q in Q])
relpos = np.array([q.answers[0][1] / max(len(ptxt[q.parent]), 1) for q in Q])
nans = np.array([len({a[0].lower() for a in q.answers}) for q in Q])
qt = Counter(QTYPES[qtype(q.text)] for q in Q)

# recouvrement lexical question / paragraphe source (part des termes de la question présents)
cov, covA = [], []
for q in Q:
    qs = set(tokenize(q.text, stem=True))
    ps = set(tokenize(ptxt[q.parent], stem=True))
    cov.append(len(qs & ps) / max(len(qs), 1))
cov = np.array(cov)

split_cnt = pd.DataFrame({
    "split": ["train", "val", "test"],
    "questions": [sum(q.split == s for q in Q) for s in ("train", "val", "test")],
    "articles": [len({q.article for q in Q if q.split == s}) for s in ("train", "val", "test")],
    "paragraphes": [len({q.parent for q in Q if q.split == s}) for s in ("train", "val", "test")],
})
split_cnt.to_csv(RES / "eda_splits.csv", index=False)

stats = {
    "n_articles": len(T), "n_paragraphes": len(P), "n_questions": len(Q),
    "mots_par_paragraphe": {"moy": float(plen.mean()), "med": float(np.median(plen)), "min": int(plen.min()), "max": int(plen.max())},
    "mots_par_question": {"moy": float(qlen.mean()), "med": float(np.median(qlen)), "min": int(qlen.min()), "max": int(qlen.max())},
    "mots_par_reponse": {"moy": float(alen.mean()), "med": float(np.median(alen)), "max": int(alen.max())},
    "questions_types": dict(qt),
    "recouvrement_question_paragraphe": {"moy": float(cov.mean()), "med": float(np.median(cov)), "part_sous_50pct": float((cov < 0.5).mean())},
    "reponses_distinctes_moy": float(nans.mean()),
    "questions_sans_recouvrement": float((cov == 0).mean()),
}
(RES / "eda_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
ax[0, 0].hist(plen, bins=40, color=GREEN); ax[0, 0].set_title("Longueur des paragraphes (mots)")
ax[0, 0].axvline(plen.mean(), color=ACCENT, ls="--", label=f"moyenne = {plen.mean():.0f}"); ax[0, 0].legend(frameon=False)
ax[0, 1].hist(qlen, bins=range(2, 36), color=GREEN); ax[0, 1].set_title("Longueur des questions (mots)")
ax[0, 1].axvline(qlen.mean(), color=ACCENT, ls="--", label=f"moyenne = {qlen.mean():.1f}"); ax[0, 1].legend(frameon=False)
names = [t for t, _ in qt.most_common()]
ax[1, 0].barh(names[::-1], [qt[n] for n in names][::-1], color=GREEN); ax[1, 0].set_title("Types de questions")
ax[1, 1].hist(cov, bins=20, color=GREEN); ax[1, 1].set_title("Recouvrement lexical question / paragraphe source")
ax[1, 1].set_xlabel("part des termes de la question présents dans le paragraphe")
for a in ax.ravel():
    a.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG / "eda_overview.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(relpos, bins=20, color=GREEN); ax[0].set_title("Position relative de la réponse dans le paragraphe")
ax[0].set_xlabel("0 = début, 1 = fin")
ax[1].hist(alen, bins=range(1, 21), color=GREY); ax[1].set_title("Longueur de la réponse (mots)")
plt.tight_layout(); plt.savefig(FIG / "eda_answers.png"); plt.close()

fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.bar(split_cnt["split"], split_cnt["questions"], color=[GREEN, GREY, ACCENT])
for i, v in enumerate(split_cnt["questions"]):
    ax.text(i, v + 60, f"{v}\n({split_cnt['articles'][i]} articles)", ha="center", fontsize=8)
ax.set_title("Découpage par article (anti-fuite)"); ax.set_ylim(0, split_cnt["questions"].max() * 1.25)
plt.tight_layout(); plt.savefig(FIG / "eda_splits.png"); plt.close()
print(json.dumps(stats, indent=2, ensure_ascii=False)); print(split_cnt)
