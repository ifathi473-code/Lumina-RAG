"""Lumina QA : recherche documentaire et réponses sourcées (BM25, fusion, reranking appris, lecture extractive).
Lancer :  streamlit run app/streamlit_app.py"""
import hashlib
import html
import io
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragx.data import load_squad  # noqa: E402
from ragx.service import RagService  # noqa: E402

DATA = ROOT / "data" / "squad_dev-v1.1.json"
GLOVE = ROOT / "data" / "glove-wiki-gigaword-100.gz"
MODEL = ROOT / "models" / "lgbm_reranker.txt"
LOGO = ROOT / "assets" / "lumina_logo.png"
ICON = ROOT / "assets" / "lumina_icon.png"
RES = ROOT / "results"
DB = ROOT / "data" / "lumina.db"
GREEN, BLUE = "#00A07F", "#2F6FB2"

st.set_page_config(page_title="Lumina QA", page_icon=str(ICON) if ICON.exists() else "🔎", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1400px;}}
.answer {{border-left:6px solid {GREEN}; background:#e9f6f2; border-radius:10px; padding:14px 18px; font-size:1.06rem;}}
.step {{border:1px solid #d8e6e1; border-radius:12px; padding:9px 12px; text-align:center; background:white; font-size:.9rem;}}
.step b {{color:{GREEN}; font-size:1.05rem;}}
.gold {{border-left:6px solid {BLUE}; background:#eef4fb; border-radius:10px; padding:10px 16px; margin-top:10px;}}
.psg {{line-height:1.65; font-size:.95rem;}}
mark.q {{background:#fff1b8; padding:0 2px; border-radius:3px;}}
mark.a {{background:#bfe8d9; padding:0 2px; border-radius:3px; font-weight:600;}}
.tag-ok {{color:white; background:{GREEN}; border-radius:10px; padding:1px 9px; font-size:.8rem;}}
.tag-no {{color:white; background:#b55a30; border-radius:10px; padding:1px 9px; font-size:.8rem;}}
.hero {{color:#5F7A73; margin:-6px 0 6px 0;}}
</style>""", unsafe_allow_html=True)

if not (DATA.exists() and GLOVE.exists()):  # premier lancement (ex. hébergement cloud)
    import subprocess
    with st.spinner("Téléchargement des données ouvertes (SQuAD, GloVe), une seule fois…"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True)

HELP_MODE = ("BM25 : classement lexical. Hybride RRF : fusion des rangs BM25 et GloVe. "
             "+ reranker : les K meilleurs candidats sont réordonnés par un modèle LightGBM entraîné pour le classement.")
FEATURE_LABEL = {"bm25": "score BM25", "bm25_rel": "BM25 relatif", "glove_cos": "similarité GloVe", "lsa_cos": "similarité LSA", "tfidf_cos": "similarité TF-IDF",
                 "rank_bm25": "rang BM25", "rank_glove": "rang GloVe", "rank_lsa": "rang LSA", "rrf": "score RRF", "coverage": "couverture des termes",
                 "idf_coverage": "couverture pondérée IDF", "bigram_match": "bigrammes communs", "bigram_frac": "part de bigrammes", "proximity": "proximité des termes",
                 "first_match_pos": "position du 1er terme", "best_sent_cov": "meilleure phrase", "len_passage": "longueur du passage", "len_query": "longueur de la question",
                 "n_numbers": "nombres", "n_years": "années", "n_caps": "mots capitalisés"}


# ------------------------------------------------------------------ données et services
def read_uploads(files):
    parts = []
    for f in files:
        name = f.name.lower()
        raw = f.getvalue()
        if name.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                parts.append("\n\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(raw)).pages))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Lecture impossible pour {f.name} : {e}")
        else:
            parts.append(raw.decode("utf8", errors="ignore"))
    return "\n\n".join(parts)


def to_passages(text, max_words=140):
    """Paragraphes séparés par une ligne vide ; les paragraphes longs sont découpés par fenêtres de ~max_words mots."""
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 25:
            continue
        w = p.split()
        if len(w) <= max_words * 1.4:
            out.append(p)
        else:
            for i in range(0, len(w), max_words):
                chunk = " ".join(w[i:i + max_words])
                if len(chunk) > 25:
                    out.append(chunk)
    return out


def warm(svc):
    for m in svc.modes:
        svc.search("warm up question", m, 3)
    return svc


@st.cache_resource(show_spinner="Construction des index (BM25, GloVe, LSA, reranker)…")
def get_squad_service():
    P, _, _ = load_squad(DATA)
    texts = [p["text"] for p in P]
    return warm(RagService(texts, GLOVE, MODEL)), texts


@st.cache_resource(show_spinner="Indexation de vos documents…", max_entries=4)
def get_custom_service(key: str, paras: tuple):
    return warm(RagService(list(paras), GLOVE, MODEL)), list(paras)


@st.cache_data
def get_questions():
    _, Q, titles = load_squad(DATA)
    by_id = {q.qid: q for q in Q}
    s = pd.read_csv(ROOT / "data" / "sample_questions.csv")
    rows = []
    for _, r in s.iterrows():
        q = by_id.get(r["qid"])
        if q is not None:
            rows.append({"question": q.text, "parent": q.parent, "answer": q.answers[0][0] if q.answers else "", "article": titles[q.article], "split": q.split})
    return pd.DataFrame(rows)


def db_connect():
    if not DB.exists():  # reconstruction automatique à partir des CSV de results/
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "15_build_db.py")], check=True)
    return sqlite3.connect(DB)


def csv(name):
    """Lit une table de résultats dans la base SQLite (data/lumina.db)."""
    try:
        with db_connect() as con:
            return pd.read_sql(f'SELECT * FROM "{name}"', con)
    except Exception:  # noqa: BLE001
        p = RES / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None


@st.cache_data
def meta_table():
    with db_connect() as con:
        return pd.read_sql("SELECT * FROM meta", con)


def source_note(*names):
    """Indique d'où viennent les chiffres d'un graphique : table, base, fichier source et script producteur."""
    m = meta_table().set_index("table_name")
    parts = []
    for n in names:
        if n in m.index:
            r = m.loc[n]
            parts.append(f"table `{n}` ← `{r['source_file']}` ({r['script']})")
    st.caption("🗄️ Source des données : base `data/lumina.db`, " + " ; ".join(parts) + ". Détail dans l'onglet **Base de données**.")


def log_search(q, mode, k, top_id, ms):
    try:
        with db_connect() as con:
            con.execute("INSERT INTO searches (question, mode, k, top_passage, total_ms) VALUES (?,?,?,?,?)", (q, mode, k, top_id, ms))
    except Exception:  # noqa: BLE001
        pass


def recent_searches(n=30):
    try:
        with db_connect() as con:
            return pd.read_sql("SELECT id, ts, question, mode, k, top_passage, total_ms FROM searches ORDER BY id DESC LIMIT ?", con, params=(n,))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["id", "ts", "question", "mode", "k", "top_passage", "total_ms"])


def bar_h(df, y, x, title_x, color=GREEN, highlight=None, fmt=".3f", height=None, sort="-x"):
    """Barres horizontales étiquetées, libellés complets, tri décroissant."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=340, labelFontSize=12)),
        x=alt.X(f"{x}:Q", title=title_x, scale=alt.Scale(domainMin=0)))
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.condition(alt.datum[y] == highlight, alt.value("#b55a30"), alt.value(color)) if highlight else alt.value(color),
        tooltip=[y, alt.Tooltip(x, format=fmt)])
    txt = base.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text(f"{x}:Q", format=fmt))
    return (bars + txt).properties(height=height or max(120, 30 * len(df)))


def terms_of(query):
    from ragx.text import tokenize
    return set(tokenize(query))


def render_passage(text, query, answer="", mark=True):
    out = html.escape(text)
    if mark:
        terms = terms_of(query)
        out = re.sub(r"\b(\w+)\b", lambda m: f'<mark class="q">{m.group(1)}</mark>' if m.group(1).lower() in terms else m.group(1), out)
    if answer and html.escape(answer) in out:
        a = html.escape(answer)
        out = out.replace(a, f'<mark class="a">{a}</mark>')
    return f'<div class="psg">{out}</div>'


def gold_rank(passages, parent):
    for r, p in enumerate(passages, 1):
        if p["id"] == parent:
            return r
    return None


def set_question(text):
    st.session_state["q"] = text


# ------------------------------------------------------------------ barre latérale
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width="stretch")
    st.markdown("#### Source des documents")
    source = st.radio("Source", ["Collection SQuAD (2 067 paragraphes)", "Mes documents"], label_visibility="collapsed")
    is_squad = source.startswith("Collection")
    custom_text = ""
    if not is_squad:
        files = st.file_uploader("Fichiers .txt, .md ou .pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)
        pasted = st.text_area("… ou collez du texte (paragraphes séparés par une ligne vide)", height=160)
        custom_text = (read_uploads(files) if files else "") + "\n\n" + pasted
    st.markdown("#### Réglages")
    top = st.slider("Passages affichés", 1, 10, 3)
    K = st.slider("Candidats du reranker (K)", 10, 100, 50, 10, help="Nombre de candidats du premier étage que le reranker réordonne. K ≈ 30 suffit sur SQuAD.")
    hl = st.toggle("Surligner les termes de la question", True)
    st.markdown("#### Historique")
    hist_df = recent_searches(40)
    if len(hist_df):
        uniq = hist_df.drop_duplicates("question").head(6)
        for i, r in enumerate(uniq.itertuples()):
            st.button(f"↺ {r.question[:34]}{'…' if len(r.question) > 34 else ''}", key=f"h{i}_{r.id}", on_click=set_question, args=(r.question,), width="stretch")
        st.caption("Enregistré dans la base (table `searches`).")
        if st.button("Vider l'historique"):
            with db_connect() as con:
                con.execute("DELETE FROM searches")
            st.rerun()
    else:
        st.caption("Vos recherches s'afficheront ici.")

st.markdown('<div class="hero">Posez une question à vos documents : les passages sont retrouvés, réordonnés et sourcés, avec le détail du classement.</div>', unsafe_allow_html=True)

if is_squad:
    svc, texts = get_squad_service()
    samples = get_questions()
else:
    paras = to_passages(custom_text)
    if len(paras) < 2:
        st.info("Ajoutez au moins deux paragraphes (fichiers ou texte collé) dans la barre latérale pour construire l'index.")
        st.stop()
    svc, texts = get_custom_service(hashlib.md5("\n".join(paras).encode()).hexdigest(), tuple(paras))
    samples = None
    st.sidebar.success(f"{len(paras)} passages indexés")


# ------------------------------------------------------------------ question
DEFAULT_Q = "What dynasty came after the Yuan?" if is_squad else "Which city has the tallest minaret in the world?"
if st.session_state.get("_src") != is_squad:  # changement de source : question par défaut adaptée
    st.session_state["_src"] = is_squad
    st.session_state["q"] = DEFAULT_Q
    st.session_state["_last"] = None
c1, c2 = st.columns([2, 3])
if is_squad:
    labels = ["Choisir un exemple…"] + [f"{r.question}  ·  {r.article}" for r in samples.itertuples()]
    ex = c1.selectbox("Question d'exemple", labels, key="ex",
                      on_change=lambda: st.session_state.update(q=st.session_state["ex"].split("  ·  ")[0]) if not st.session_state["ex"].startswith("Choisir") else None)
q = c2.text_input("Votre question", key="q", placeholder="Posez votre question (en anglais pour la collection SQuAD)")
gold = None
if is_squad:
    m = samples[samples["question"] == q.strip()]
    if len(m):
        gold = m.iloc[0]

tab_search, tab_cmp, tab_perf, tab_db, tab_how, tab_about = st.tabs(["🔎 Recherche", "⚖️ Comparer", "📊 Performances", "🗄️ Base de données", "🧠 Fonctionnement", "ℹ️ À propos"])

# ================================================================== RECHERCHE
with tab_search:
    mode = st.radio("Stratégie", svc.modes, horizontal=True, help=HELP_MODE, key="mode")
    if not q.strip():
        st.info("Saisissez une question.")
    else:
        res = svc.search(q, mode, max(top, 10), K=K)
        shown = res.passages[:top]
        key = (q, mode, K)
        if st.session_state.get("_last") != key:
            st.session_state["_last"] = key
            log_search(q, mode, K, int(res.passages[0]["id"]) if res.passages else -1, res.timings_ms["total_ms"])
        left, right = st.columns([3, 1.25])
        with left:
            st.markdown("#### Réponse")
            st.markdown(f'<div class="answer">{html.escape(res.answer_sentence or "Aucune phrase trouvée.")}</div>', unsafe_allow_html=True)
            if gold is not None:
                r = gold_rank(res.passages, int(gold["parent"]))
                tag = '<span class="tag-ok">bon passage en 1ʳᵉ position</span>' if r == 1 else (
                    f'<span class="tag-no">bon passage en position {r}</span>' if r else '<span class="tag-no">bon passage hors du top 10</span>')
                st.markdown(f'<div class="gold"><b>Réponse annotée :</b> {html.escape(gold["answer"])} &nbsp; {tag}<br><small>Article : {html.escape(gold["article"])}</small></div>', unsafe_allow_html=True)
            st.markdown("#### Sources")
            for r, p in enumerate(shown, 1):
                is_gold = gold is not None and p["id"] == int(gold["parent"])
                with st.expander(f"#{r} · score {p['score']:.3f} · passage {p['id']}" + ("  ✅ bon passage" if is_gold else ""), expanded=(r == 1)):
                    st.markdown(render_passage(p["text"], q, gold["answer"] if is_gold else "", hl), unsafe_allow_html=True)
                    st.caption(f"{len(p['text'].split())} mots · {len(terms_of(q) & terms_of(p['text']))} termes de la question présents")
                    if mode.endswith("LightGBM"):
                        ex_ = svc.explain(q, p["id"], K)
                        if ex_ and st.toggle("Pourquoi ce score ? (contributions SHAP)", key=f"why{r}_{p['id']}"):
                            d = pd.DataFrame([(FEATURE_LABEL.get(n, n.replace("qtype_", "question : ")), c) for n, v, c in ex_[:8]], columns=["variable", "contribution"]).set_index("variable")
                            st.bar_chart(d, color=BLUE, horizontal=True, height=260)
                            st.caption("Positif : la variable pousse le passage vers le haut du classement ; négatif : vers le bas.")
            dl1, dl2, _ = st.columns([1, 1, 3])
            payload = {"question": q, "strategie": mode, "K": K, "reponse": res.answer_sentence, "passages": res.passages[:top], "latence_ms": res.timings_ms}
            dl1.download_button("⬇ JSON", json.dumps(payload, ensure_ascii=False, indent=2), "resultat_lumina.json", "application/json")
            dl2.download_button("⬇ CSV", pd.DataFrame(res.passages[:top]).to_csv(index=False), "passages_lumina.csv", "text/csv")
        with right:
            st.markdown("#### Pipeline")
            t = res.timings_ms
            for k, lab in [("1er_etage_ms", "1 · Premier étage"), ("reranking_ms", "2 · Reranking"), ("lecture_ms", "3 · Lecture")]:
                if k in t:
                    st.markdown(f'<div class="step">{lab}<br><b>{t[k]:.1f} ms</b></div><div style="text-align:center;color:#aaa;line-height:1">▼</div>', unsafe_allow_html=True)
            st.metric("Temps total", f"{t['total_ms']:.1f} ms")
            h = recent_searches(15).iloc[::-1]
            if len(h) > 1:
                st.caption("Latence des dernières recherches (ms), lue dans la table `searches`")
                st.altair_chart(alt.Chart(h.reset_index(drop=True).reset_index()).mark_line(point=True, color=GREEN).encode(
                    x=alt.X("index:Q", title="recherche n°", axis=alt.Axis(tickMinStep=1)), y=alt.Y("total_ms:Q", title="ms"), tooltip=["question", "mode", "total_ms"]).properties(height=150), width="stretch")
            st.caption("Mesuré en laboratoire : 8,0 ms (médiane) et 12,2 ms (p95) sur 1 cœur.")

# ================================================================== COMPARER
with tab_cmp:
    st.markdown("Les stratégies sur **la même question** : où se place le bon passage, et quelle phrase est retenue ?")
    if not q.strip():
        st.info("Saisissez une question.")
    else:
        cols = st.columns(len(svc.modes))
        summary = []
        for col, m in zip(cols, svc.modes):
            r = svc.search(q, m, 10, K=K)
            gr = gold_rank(r.passages, int(gold["parent"])) if gold is not None else None
            summary.append({"stratégie": m, "rang du bon passage": gr if gr else ("hors top 10" if gold is not None else "n/a"), "temps (ms)": round(r.timings_ms["total_ms"], 1),
                            "passage n°1": r.passages[0]["id"]})
            with col:
                st.markdown(f"##### {m}")
                st.markdown(f'<div class="answer" style="font-size:.9rem">{html.escape(r.answer_sentence or "–")}</div>', unsafe_allow_html=True)
                for i, p in enumerate(r.passages[:3], 1):
                    mark = " ✅" if gold is not None and p["id"] == int(gold["parent"]) else ""
                    with st.expander(f"#{i} · passage {p['id']} · {p['score']:.3f}{mark}"):
                        st.markdown(render_passage(p["text"], q, mark=hl), unsafe_allow_html=True)
        st.markdown("##### Synthèse")
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
        if is_squad:
            st.markdown("---")
            st.markdown("##### Mini-évaluation en direct")
            n = st.slider("Nombre de questions d'exemple", 10, len(samples), len(samples), 10)
            st.caption("Hit@1, Hit@3 et MRR@10 recalculés ici, sur l'échantillon (indicatif ; les chiffres de l'onglet Performances portent sur 3 061 questions).")
            if st.button("Lancer l'évaluation"):
                rows, prog = [], st.progress(0.0)
                sub = samples.head(n)
                for mi, m in enumerate(svc.modes):
                    h1 = h3 = mrr = 0.0
                    for qi, r_ in enumerate(sub.itertuples()):
                        rr = gold_rank(svc.search(r_.question, m, 10, K=K).passages, int(r_.parent))
                        h1 += rr == 1
                        h3 += bool(rr and rr <= 3)
                        mrr += 1 / rr if rr else 0
                        prog.progress((mi * n + qi + 1) / (len(svc.modes) * n))
                    rows.append({"stratégie": m, "Hit@1": round(h1 / n, 3), "Hit@3": round(h3 / n, 3), "MRR@10": round(mrr / n, 3)})
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, width="stretch")
                st.bar_chart(df.set_index("stratégie")[["Hit@1", "Hit@3", "MRR@10"]], stack=False, height=260)
                st.download_button("⬇ Résultats CSV", df.to_csv(index=False), "evaluation_lumina.csv", "text/csv")

# ================================================================== PERFORMANCES
with tab_perf:
    st.markdown("### Évaluation sur 3 061 questions (16 articles jamais vus à l'entraînement)")
    rr = csv("rerank_results")
    rc = csv("retrieval_by_chunking")
    if rc is not None and rr is not None:
        names = {"bm25": "BM25 (sans racinisation)", "bm25_stem": "BM25 + Porter", "tfidf": "TF-IDF", "lsa": "LSA (SVD 256)", "glove": "GloVe-SIF",
                 "hybrid_rrf": "Hybride RRF", "hybrid_wsum": "Hybride pondéré (α = 0,7)"}
        t1 = rc[rc["gran"] == "para"].copy()
        t1["système"] = t1["retriever"].map(names)
        t1 = t1.dropna(subset=["système"])[["système", "hit@1", "mrr@10", "ndcg@10"]]
        rn = {"lgbm": "Hybride + reranker LightGBM", "logreg": "Hybride + reranker logistique", "oracle (borne sup.)": "Oracle (borne supérieure)"}
        t2 = rr[(rr["gran"] == "para") & rr["reranker"].isin(rn)].copy()
        t2["système"] = t2["reranker"].map(rn)
        tab = pd.concat([t1, t2[["système", "hit@1", "mrr@10", "ndcg@10"]]]).round(3)
        val = lambda n_: float(tab[tab["système"] == n_]["hit@1"].iloc[0])  # noqa: E731
        base = val("BM25 + Porter")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("BM25 + Porter · Hit@1", f"{base:.3f}")
        k2.metric("Hybride RRF · Hit@1", f"{val('Hybride RRF'):.3f}", f"{(val('Hybride RRF') - base) * 100:+.1f} pts")
        k3.metric("+ reranker LightGBM · Hit@1", f"{val('Hybride + reranker LightGBM'):.3f}", f"{(val('Hybride + reranker LightGBM') - base) * 100:+.1f} pts")
        k4.metric("Oracle (borne sup.)", f"{val('Oracle (borne supérieure)'):.3f}")
        cl, cr = st.columns([2, 3])
        cl.dataframe(tab, hide_index=True, width="stretch", height=430)
        cr.markdown("**Hit@1 par système** (part des questions dont le bon passage est classé premier)")
        cr.altair_chart(bar_h(tab, "système", "hit@1", "Hit@1", highlight="Hybride + reranker LightGBM", height=400), width="stretch")
        source_note("retrieval_by_chunking", "rerank_results")
        st.caption("Orange : meilleur système réel. Intervalles de confiance à 95 % (bootstrap par articles) et tests appariés : table `stat_tests`.")

    st.markdown("---")
    c1_, c2_ = st.columns(2)
    k_ = rr[(rr["gran"] == "para") & rr["reranker"].isin(["lgbm_K", "logreg_K"])] if rr is not None else None
    if k_ is not None and len(k_):
        kk = k_.assign(modèle=k_["reranker"].map({"lgbm_K": "LightGBM", "logreg_K": "Régression logistique"}))[["K", "modèle", "mrr@10"]]
        c1_.markdown("#### Combien de candidats faut-il reranker ? (MRR@10 selon K)")
        c1_.altair_chart(alt.Chart(kk).mark_line(point=True).encode(x=alt.X("K:Q", title="K (candidats réordonnés)", scale=alt.Scale(domainMin=0)),
                         y=alt.Y("mrr@10:Q", title="MRR@10", scale=alt.Scale(zero=False)), color=alt.Color("modèle:N", scale=alt.Scale(range=[BLUE, GREEN]), title=None),
                         tooltip=["K", "modèle", alt.Tooltip("mrr@10", format=".3f")]).properties(height=280), width="stretch")
        with c1_:
            source_note("rerank_results")
    g = csv("perm_importance_groups")
    if g is not None:
        g = g.assign(groupe_court=g["groupe"].str.split(" \\(").str[0])
        c2_.markdown("#### Quelles variables comptent ? (perte de MRR par permutation)")
        c2_.altair_chart(bar_h(g, "groupe_court", "delta_mrr", "perte de MRR quand le groupe est mélangé", color=BLUE, height=200), width="stretch")
        with c2_:
            source_note("perm_importance_groups")
    c3_, c4_ = st.columns(2)
    ec = csv("error_categories")
    if ec is not None:
        c3_.markdown("#### Où le meilleur système se trompe-t-il ?")
        ec = ec.assign(part_pct=ec["part"] * 100)
        c3_.altair_chart(bar_h(ec, "categorie", "part_pct", "% des questions de test", color=GREEN, fmt=".1f", height=190, sort=None), width="stretch")
        with c3_:
            source_note("error_categories")
    ov = csv("error_by_overlap")
    if ov is not None:
        c4_.markdown("#### Réussite selon le recouvrement lexical question-passage")
        c4_.altair_chart(alt.Chart(ov).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("couverture_lexicale:N", sort=None, title="part des mots de la question présents dans le passage", axis=alt.Axis(labelAngle=0)), y=alt.Y("hit@1:Q", title="Hit@1", scale=alt.Scale(domain=[0, 1])),
            tooltip=["couverture_lexicale", alt.Tooltip("hit@1", format=".3f"), "effectif"]).properties(height=190), width="stretch")
        with c4_:
            source_note("error_by_overlap")
    c5_, c6_ = st.columns(2)
    sh = csv("shap_importance")
    if sh is not None:
        num = sh.select_dtypes("number").columns[0]
        namec = [c for c in sh.columns if c not in sh.select_dtypes("number").columns][0] if len(sh.select_dtypes("number").columns) < len(sh.columns) else sh.columns[0]
        top10 = sh.sort_values(num, ascending=False).head(10).assign(**{namec: lambda d: d[namec].map(lambda v: FEATURE_LABEL.get(v, v))})
        c5_.markdown("#### Les 10 variables les plus influentes (SHAP moyen)")
        c5_.altair_chart(bar_h(top10, namec, num, "|valeur SHAP| moyenne", color=GREEN, fmt=".3f", height=300), width="stretch")
        with c5_:
            source_note("shap_importance")
    lat = csv("latency")
    if lat is not None:
        row = lat[lat["K"] == lat["K"].max()].iloc[0]
        stages = pd.DataFrame({"étage": ["BM25", "GloVe", "LSA", "Fusion + variables", "LightGBM", "Lecture"],
                               "ms": [row["bm25_p50"], row["glove_p50"], row["lsa_p50"], row["fusion+features_p50"], row["lgbm_p50"], row["reader_p50"]]})
        c6_.markdown(f"#### Où part le temps ? (médiane, K = {int(row['K'])}, total {row['total_p50']:.1f} ms)")
        c6_.altair_chart(bar_h(stages, "étage", "ms", "millisecondes par requête", color=BLUE, fmt=".2f", height=300), width="stretch")
        with c6_:
            source_note("latency")
    figs = [("rerank_effect.png", "Effet du reranker"), ("shap_summary.png", "Résumé SHAP"), ("k_sensitivity.png", "Sensibilité à K")]
    st.markdown("#### Figures produites par les scripts")
    for col, (f, cap) in zip(st.columns(len(figs)), figs):
        if (RES / "figures" / f).exists():
            col.image(str(RES / "figures" / f), caption=f"{cap} (results/figures/{f})", width="stretch")

# ================================================================== BASE DE DONNÉES
with tab_db:
    st.markdown("### Où sont les données ?")
    st.markdown("Tous les chiffres et graphiques de l'application sont lus dans une base **SQLite** (`data/lumina.db`), reconstruite par `python scripts/15_build_db.py` à partir des fichiers produits par les scripts d'expériences.")
    st.graphviz_chart("""
    digraph L { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#e9f6f2", color="#00A07F"];
      S [label="SQuAD v1.1 dev\\n(data/squad_dev-v1.1.json)\\nGloVe 100d", shape=cylinder, fillcolor="#eef4fb", color="#2f6fb2"];
      X [label="Scripts 01 à 10\\n(EDA, expériences,\\ninterprétabilité, latence)"];
      C [label="results/*.csv\\nresults/figures/*.png", shape=note];
      B [label="data/lumina.db\\n(SQLite, {NT} tables)", shape=cylinder, fillcolor="#fdeee6", color="#b55a30"];
      A [label="Application Lumina QA\\nPerformances, Comparer…"];
      H [label="table searches\\n(historique)", shape=cylinder, fillcolor="#fdeee6", color="#b55a30"];
      S -> X; X -> C; C -> B [label="15_build_db.py"]; B -> A; A -> H [label="écrit"]; H -> A [label="lit"]; S -> A [label="index de recherche"]; }
    """.replace("{NT}", str(len(meta_table()))))
    meta = meta_table()
    st.markdown("#### Catalogue des tables")
    st.dataframe(meta.rename(columns={"table_name": "table", "source_file": "fichier source", "script": "script producteur", "rows": "lignes"}), hide_index=True, width="stretch", height=320)
    c1d, c2d = st.columns([1, 3])
    tname = c1d.selectbox("Explorer une table", meta["table_name"].tolist(), index=meta["table_name"].tolist().index("rerank_results") if "rerank_results" in meta["table_name"].tolist() else 0)
    with db_connect() as con:
        cols = pd.read_sql(f'PRAGMA table_info("{tname}")', con)
        data = pd.read_sql(f'SELECT * FROM "{tname}" LIMIT 500', con)
    c1d.markdown("**Colonnes**")
    c1d.dataframe(cols[["name", "type"]].rename(columns={"name": "colonne", "type": "type"}), hide_index=True, width="stretch")
    c2d.markdown(f"**Contenu de `{tname}`** (500 premières lignes)")
    c2d.dataframe(data, hide_index=True, width="stretch", height=330)
    st.markdown("#### Requête SQL (lecture seule)")
    sql = st.text_area("SELECT …", "SELECT retriever, ROUND(\"hit@1\", 3) AS hit1, ROUND(\"mrr@10\", 3) AS mrr10\nFROM retrieval_by_chunking\nWHERE gran = 'para'\nORDER BY \"hit@1\" DESC;", height=110)
    if st.button("Exécuter la requête"):
        if not sql.strip().lower().startswith("select"):
            st.error("Seules les requêtes SELECT sont autorisées.")
        else:
            try:
                with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as ro:
                    out = pd.read_sql(sql, ro)
                st.dataframe(out, hide_index=True, width="stretch")
                st.download_button("⬇ Résultat CSV", out.to_csv(index=False), "requete_lumina.csv", "text/csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"Erreur SQL : {e}")
    st.download_button("⬇ Télécharger la base lumina.db", DB.read_bytes(), "lumina.db", "application/octet-stream")

# ================================================================== FONCTIONNEMENT
with tab_how:
    st.markdown("### Le pipeline en un schéma")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e9f6f2", color="#00A07F", fontname="Helvetica"];
      Q [label="Question"]; I [label="Index\\n(BM25, GloVe-SIF, LSA, TF-IDF)", shape=cylinder];
      F [label="Fusion RRF\\nBM25 + GloVe"]; K [label="Top-K\\ncandidats"]; R [label="Reranker LightGBM\\n(29 variables)", fillcolor="#fdeee6", color="#b55a30"];
      L [label="Lecteur extractif\\n(meilleure phrase)"]; A [label="Réponse +\\nsources", fillcolor="#eef4fb", color="#2f6fb2"];
      Q -> F; I -> F [style=dashed, color="#2f6fb2"]; F -> K; K -> R; R -> L; L -> A; }
    """)
    a, b = st.columns(2)
    with a:
        st.markdown("#### 1. Premier étage : retrouver vite")
        st.latex(r"\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{IDF}(t)\,\frac{\mathrm{tf}_{t,d}(k_1+1)}{\mathrm{tf}_{t,d}+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}")
        st.markdown("BM25 compte les mots de la question présents dans le passage, avec saturation de la fréquence et normalisation par la longueur ($k_1=1{,}2$, $b=0{,}75$). **GloVe-SIF** représente un texte par une moyenne pondérée de vecteurs de mots, comparée à la question par cosinus.")
        st.markdown("#### 2. Fusion")
        st.latex(r"\mathrm{RRF}(d)=\sum_{r\in\{\mathrm{BM25},\mathrm{GloVe}\}}\frac{1}{60+\mathrm{rang}_r(d)}")
    with b:
        st.markdown("#### 3. Reranking appris")
        st.markdown("Pour chacun des K candidats, 29 variables (scores lexicaux et sémantiques, rangs, couverture des termes, bigrammes, proximité, longueur) alimentent un modèle **LightGBM LambdaRank** qui optimise directement le nDCG. Le bouton « Pourquoi ce score ? » détaille la contribution de chaque variable (valeurs SHAP).")
        st.latex(r"\lambda_{ij}=\frac{-\sigma\,|\Delta \mathrm{nDCG}_{ij}|}{1+e^{\sigma(s_i-s_j)}}")
        st.markdown("#### 4. Lecture extractive")
        st.markdown("La phrase des trois premiers passages qui recouvre le mieux les termes de la question (pondérés par l'IDF) est renvoyée. C'est un lecteur extractif, sans génération.")
    st.markdown("#### Utiliser l'API")
    st.code("curl 'http://localhost:8000/search?q=What%20dynasty%20came%20after%20the%20Yuan%3F&mode=Hybride%20%2B%20reranker%20LightGBM&top=3&k=50'", language="bash")

# ================================================================== À PROPOS
with tab_about:
    st.markdown("### Données")
    st.markdown("""
- **Collection de démonstration : SQuAD v1.1 (dev)**, 48 articles Wikipédia, 2 067 paragraphes, 10 570 questions (CC BY-SA 4.0).
- **GloVe wiki-gigaword 100d** pour le premier étage sémantique.
- **Vos documents** : fichiers .txt, .md, .pdf ou texte collé, découpés en passages et indexés à la volée.
- Résultats et historique : base SQLite `data/lumina.db` (onglet **Base de données**). Sources, prétraitement et empreintes SHA-256 : `data/DATA.md`, `data/splits.json`, `data/checksums.sha256`.
""")
    st.markdown("### Limites")
    st.markdown("""
- Collection de démonstration en anglais ; les chiffres d'évaluation viennent d'un seul jeu de données.
- Embeddings statiques (GloVe) : des bi-encodeurs et cross-encodeurs neuronaux sont prévus en extension mais n'ont pas été exécutés.
- Lecteur extractif : pas de génération de texte ; environ 11 % des erreurs apparentes sont des fausses négatives d'annotation.
- Sur vos propres documents, le reranker (entraîné sur SQuAD) peut moins bien généraliser : comparez avec BM25 dans l'onglet **Comparer**.
""")
    st.markdown("### Technologies")
    st.markdown("Python, scikit-learn, LightGBM, SHAP, NLTK, Streamlit, FastAPI, Docker.")
    if is_squad:
        st.markdown("### Questions d'exemple")
        st.dataframe(samples[["question", "answer", "article"]], hide_index=True, width="stretch", height=260)
