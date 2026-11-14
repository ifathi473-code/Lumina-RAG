"""Statistiques complémentaires sur les erreurs du reranker (même article ? réponse tout de même présente ?)."""
import json
import pickle

import numpy as np

from common import DATA, RES
from ragx.data import build_qrels, load_squad, make_passages

C = pickle.load(open(RES / "cache" / "rerank_para.pkl", "rb"))
P, Q, T = load_squad(DATA)
ps = make_passages(P)
qr = build_qrels(Q, ps)
qs = [q for q in Q if q.split == "test"]
lg = C["lgbm_ranking"]
err = [i for i, q in enumerate(qs) if lg[i][0] not in qr[q.qid]]
same_art = np.mean([ps[lg[i][0]].article == qs[i].article for i in err])
has_ans = np.mean([any(a.lower() in ps[lg[i][0]].text.lower() for a, _ in qs[i].answers) for i in err])
out = {"n_test": len(qs), "n_errors_top1": len(err), "part_erreurs_meme_article": float(same_art),
       "part_erreurs_reponse_presente_dans_top1": float(has_ans)}
(RES / "error_stats.json").write_text(json.dumps(out, indent=2))
print(out)
