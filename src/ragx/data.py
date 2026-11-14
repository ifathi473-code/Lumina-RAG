"""Chargement de SQuAD v1.1 (dev), découpage par article (anti-fuite) et chunking."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLIT_SEED = 42


@dataclass
class Passage:
    pid: int            # identifiant de passage (chunk) dans l'index
    parent: int         # identifiant du paragraphe d'origine
    article: int
    text: str
    cstart: int         # offset caractère dans le paragraphe
    cend: int


@dataclass
class Question:
    qid: str
    text: str
    article: int
    parent: int         # paragraphe contenant la réponse
    answers: list[tuple[str, int]] = field(default_factory=list)  # (texte, start)
    split: str = "test"


def load_squad(path: str | Path):
    """Retourne (paragraphes, questions, titres). Split par ARTICLE : train/val/test."""
    data = json.loads(Path(path).read_text(encoding="utf8"))["data"]
    paragraphs: list[dict] = []
    questions: list[Question] = []
    titles = [a["title"] for a in data]
    for ai, art in enumerate(data):
        for par in art["paragraphs"]:
            pi = len(paragraphs)
            paragraphs.append({"parent": pi, "article": ai, "text": par["context"]})
            for qa in par["qas"]:
                answers = [(a["text"], a["answer_start"]) for a in qa["answers"]]
                questions.append(Question(qa["id"], qa["question"], ai, pi, answers))
    # --- split par article, fixé par graine : aucune question d'un même article
    # n'apparaît dans deux jeux différents.
    rng = np.random.RandomState(SPLIT_SEED)
    order = rng.permutation(len(titles))
    n = len(titles)
    n_train, n_val = n // 2, n // 6
    split_of = {}
    for rank, ai in enumerate(order):
        split_of[int(ai)] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    for q in questions:
        q.split = split_of[q.article]
    return paragraphs, questions, titles


def make_passages(paragraphs: list[dict], size: int | None = None, stride: int | None = None) -> list[Passage]:
    """Chunking par fenêtre glissante de `size` mots (pas `stride`).
    size=None -> un passage = un paragraphe entier."""
    out: list[Passage] = []
    for par in paragraphs:
        text = par["text"]
        if size is None:
            out.append(Passage(len(out), par["parent"], par["article"], text, 0, len(text)))
            continue
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        stride_ = stride or size
        i = 0
        while True:
            j = min(i + size, len(spans))
            cs, ce = spans[i][0], spans[j - 1][1]
            out.append(Passage(len(out), par["parent"], par["article"], text[cs:ce], cs, ce))
            if j >= len(spans):
                break
            i += stride_
    return out


def relevant_set(q: Question, passages_by_parent: dict[int, list[Passage]]) -> set[int]:
    """Passages pertinents : ceux du paragraphe source qui CONTIENNENT entièrement
    au moins une réponse annotée (vérité terrain exacte, par offsets)."""
    rel = set()
    for p in passages_by_parent[q.parent]:
        for txt, st in q.answers:
            if st >= p.cstart and st + len(txt) <= p.cend:
                rel.add(p.pid)
                break
    return rel


def build_qrels(questions: list[Question], passages: list[Passage]) -> dict[str, set[int]]:
    by_parent: dict[int, list[Passage]] = {}
    for p in passages:
        by_parent.setdefault(p.parent, []).append(p)
    return {q.qid: relevant_set(q, by_parent) for q in questions}
