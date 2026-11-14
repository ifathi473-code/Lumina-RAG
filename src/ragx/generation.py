"""Génération (étage G du RAG) : prompt ancré, générateur interchangeable, métriques EM / F1,
et prompt de « LLM-juge » pour la fidélité. Les générateurs neuronaux sont optionnels."""
from __future__ import annotations

import re
from collections import Counter

from .text import normalize_answer

PROMPT = (
    "Réponds à la question en utilisant UNIQUEMENT le contexte. "
    "Si la réponse n'y figure pas, réponds « je ne sais pas ». Réponse courte.\n\n"
    "Contexte :\n{context}\n\nQuestion : {question}\nRéponse :"
)
PROMPT_EN = (
    "Answer the question using ONLY the context. If the answer is not in the context, say 'I don't know'. "
    "Give a short answer.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)
JUDGE = (
    "You are grading a RAG answer. Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
    "1) FAITHFULNESS: is every claim in the answer supported by the context? Reply 1 (yes) or 0 (no).\n"
    "2) RELEVANCE: does the answer address the question? Reply 1 or 0.\nReply as JSON: {{\"faithfulness\": x, \"relevance\": y}}"
)


def build_prompt(question: str, passages: list[str], english: bool = True) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return (PROMPT_EN if english else PROMPT).format(context=ctx, question=question)


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def f1_score(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = normalize_answer(pred).split()
    for g in golds:
        gt = normalize_answer(g).split()
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        pr, rc = n / len(p), n / len(gt)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


class HFGenerator:
    """Générateur seq2seq Hugging Face (par défaut flan-t5-base)."""

    def __init__(self, name: str = "google/flan-t5-base", max_new_tokens: int = 24):
        from transformers import pipeline
        self.pipe = pipeline("text2text-generation", model=name)
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        return self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)[0]["generated_text"].strip()


def parse_judge(text: str) -> tuple[float, float]:
    f = re.search(r'"faithfulness"\s*:\s*([01])', text)
    r = re.search(r'"relevance"\s*:\s*([01])', text)
    return (float(f.group(1)) if f else float("nan"), float(r.group(1)) if r else float("nan"))
