"""ÉVALUATION DE LA GÉNÉRATION (optionnelle) : flan-t5 (ou autre) sur N questions de test,
EM / F1 SQuAD pour 3 systèmes de contexte. Nécessite accès à Hugging Face.
    pip install transformers sentencepiece torch
    python scripts/06_generation_optional.py --n 300"""
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from common import DATA, RES, ROOT
from ragx.data import build_qrels, load_squad, make_passages
from ragx.generation import HFGenerator, build_prompt, exact_match, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--model", default="google/flan-t5-base")
ap.add_argument("--ctx", type=int, default=3)
a = ap.parse_args()

P, Q, T = load_squad(DATA)
passages = make_passages(P)
qrels = build_qrels(Q, passages)
qs = [q for q in Q if q.split == "test"][: a.n]
C = pickle.load(open(RES / "cache" / "rerank_para.pkl", "rb"))
pos = {qid: i for i, qid in enumerate(C["qids_test"])}
gen = HFGenerator(a.model)
rows = []
systems = {
    "Hybride (RRF), sans reranking": lambda i, q: C["cand_test"][i][: a.ctx],
    "Hybride + LightGBM": lambda i, q: C["lgbm_ranking"][i][: a.ctx],
    "Oracle (bon paragraphe)": lambda i, q: [next(iter(qrels[q.qid]))],
}
for name, pick in systems.items():
    em, f1 = [], []
    for q in qs:
        ids = pick(pos[q.qid], q)
        pred = gen(build_prompt(q.text, [passages[p].text for p in ids]))
        golds = [t for t, _ in q.answers]
        em.append(exact_match(pred, golds)); f1.append(f1_score(pred, golds))
    rows.append({"systeme": name, "EM": np.mean(em), "F1": np.mean(f1)})
df = pd.DataFrame(rows)
df.to_csv(RES / "generation_results.csv", index=False)
tex = ["\\begin{tabular}{lrr}", "\\toprule", "Contexte fourni au générateur & EM & F1 \\\\", "\\midrule"]
for _, r in df.iterrows():
    tex.append(f"{r['systeme']} & {r['EM']:.3f} & {r['F1']:.3f} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(RES / "generation_table.tex").write_text("\n".join(tex)); (ROOT / "report" / "tables").mkdir(parents=True, exist_ok=True); (ROOT / "report" / "tables" / "generation_table.tex").write_text("\n".join(tex))
print(df.round(3))
