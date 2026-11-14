"""Prétraitement du texte : normalisation, tokenisation, stop-words, racinisation."""
from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def _stem(w: str) -> str:
    return _stemmer.stem(w)


def tokenize(text: str, stop: bool = True, stem: bool = False) -> list[str]:
    """Minuscule -> tokens alphanumériques -> (stop-words) -> (Porter)."""
    toks = _TOKEN_RE.findall(text.lower())
    toks = [t.split("'")[0] for t in toks]
    if stop:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [_stem(t) for t in toks]
    return toks


def normalize_answer(s: str) -> str:
    """Normalisation de type SQuAD : minuscules, sans ponctuation ni articles."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p for p in parts if p]
