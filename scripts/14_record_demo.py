"""Enregistre la vidéo de démonstration (Playwright + ffmpeg) à partir de l'application en cours d'exécution.

Prérequis : l'app Streamlit tourne sur :8599 et l'API sur :8000, par exemple
    streamlit run app/streamlit_app.py --server.port 8599 --server.headless true &
    uvicorn app.api:app --port 8000 &
Usage : python scripts/14_record_demo.py   ->  demo/demo_plateforme_RAG.mp4
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)
APP, API = "http://localhost:8599", "http://localhost:8000"
W, H = 1440, 900

OVERLAY = """
(() => {
 if (document.getElementById('cap')) return;
 const c = document.createElement('div'); c.id = 'cap';
 c.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);max-width:1100px;z-index:99999;background:rgba(20,40,36,.92);color:#fff;font:600 21px Helvetica,Arial,sans-serif;padding:13px 26px;border-radius:12px;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,.35);transition:opacity .3s';
 document.body.appendChild(c);
 const d = document.createElement('div'); d.id = 'cur';
 d.style.cssText = 'position:fixed;width:22px;height:22px;border-radius:50%;background:rgba(255,120,0,.55);border:2px solid #ff7800;z-index:99999;pointer-events:none;left:-50px;top:-50px;transition:left .02s,top .02s';
 document.body.appendChild(d);
 document.addEventListener('mousemove', e => { d.style.left = (e.clientX - 11) + 'px'; d.style.top = (e.clientY - 11) + 'px'; });
})();
"""


def main():
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        b = p.chromium.launch()
        # échauffement hors enregistrement : charge l'index et chauffe les trois stratégies
        wp = b.new_page(viewport={"width": W, "height": H})
        wp.goto(APP)
        wp.wait_for_selector("text=Pipeline", timeout=120000)
        for lab in ["BM25", "Hybride (RRF)", "Hybride + reranker", "BM25"]:
            wp.locator("label", has_text=lab).first.click()
            wp.wait_for_timeout(2500)
        wp.close()
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
        pg = ctx.new_page()

        def cap(text, secs=3.0):
            pg.evaluate(OVERLAY)
            pg.evaluate("t => { const c = document.getElementById('cap'); c.textContent = t; c.style.opacity = t ? 1 : 0; }", text)
            pg.wait_for_timeout(int(secs * 1000))

        def move_click(loc):
            loc.scroll_into_view_if_needed()
            bb = loc.bounding_box()
            pg.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2, steps=30)
            pg.wait_for_timeout(250)
            pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)

        def scroll(total, step=60, pause=40):
            for _ in range(total // step):
                pg.mouse.wheel(0, step)
                pg.wait_for_timeout(pause)

        def top():
            pg.evaluate("document.querySelectorAll('[data-testid=stMain], section.main, .main').forEach(e => e.scrollTo({top: 0}))")
            pg.wait_for_timeout(900)

        def tab(name):
            top()
            move_click(pg.locator("[role=tab]", has_text=name))
            pg.wait_for_timeout(2500)

        def strategy(label):
            top()
            move_click(pg.locator("label", has_text=label).first)
            pg.wait_for_timeout(2800)

        pg.goto(APP)
        pg.wait_for_selector("text=Pipeline", timeout=120000)
        pg.mouse.move(700, 450)
        pg.wait_for_timeout(1500)

        # 1. Présentation
        cap("Lumina QA : recherche documentaire, reranking et réponses sourcées", 4)
        cap("À gauche : source des documents, réglages (passages, candidats K) et historique", 3.5)

        # 2. Recherche BM25 sur une question où BM25 se trompe
        cap("Question SQuAD : « What dynasty came after the Yuan? » avec BM25 seul", 3.5)
        cap("La réponse annotée est affichée, avec la position du bon passage : ici seulement 3ᵉ", 4.5)
        move_click(pg.locator("[data-testid=stExpander] summary").nth(1))
        pg.wait_for_timeout(800)
        scroll(240)
        cap("Les mots de la question sont surlignés en jaune dans chaque passage", 3.5)
        top()

        # 3. Autres stratégies
        strategy("Hybride (RRF)")
        cap("Hybride RRF (BM25 + GloVe) : le bon passage passe en 1ʳᵉ position", 4)
        strategy("Hybride + reranker")
        cap("Avec le reranker LightGBM : le bon passage reste en tête, latence détaillée par étage", 4)
        move_click(pg.locator("label", has_text="Pourquoi ce score").first)
        pg.wait_for_timeout(2500)
        scroll(420)
        cap("« Pourquoi ce score ? » : contribution de chaque variable (valeurs SHAP) au classement", 5)
        top()
        cap("Le pipeline complet prend quelques millisecondes par requête", 3)

        # 4. Comparer
        tab("Comparer")
        cap("Onglet « Comparer » : les trois stratégies sur la même question", 3.5)
        scroll(300)
        cap("Rang du bon passage et temps total pour chaque stratégie", 3.5)
        mini = pg.locator("button", has_text="Lancer l'évaluation").first
        mini.wait_for(state="attached", timeout=60000)
        n_before = pg.evaluate("document.querySelectorAll('[data-testid=stVegaLiteChart], [data-testid=stArrowVegaLiteChart]').length")
        move_click(mini)
        pg.wait_for_function("n => document.querySelectorAll('[data-testid=stVegaLiteChart], [data-testid=stArrowVegaLiteChart]').length > n", arg=n_before, timeout=180000)
        pg.wait_for_timeout(1200)
        scroll(400)
        cap("Mini-évaluation en direct sur l'échantillon : Hit@1, Hit@3 et MRR@10 par stratégie", 5)

        # 5. Une question où la fusion échoue
        top()
        tab("Recherche")
        sel = pg.locator("[data-testid=stSelectbox]").first
        top()
        move_click(sel)
        pg.keyboard.type("Stephen Eildmann", delay=70)
        pg.wait_for_timeout(600)
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(2500)
        tab("Comparer")
        cap("Autre cas : ici la fusion RRF à poids égaux dégrade le classement, BM25 et le reranker restent bons", 5.5)
        scroll(250)
        pg.wait_for_timeout(1500)
        top()

        # 6. Explications
        tab("Fonctionnement")
        cap("Onglet « Fonctionnement » : schéma du pipeline et formules (BM25, RRF, LambdaRank)", 4)
        scroll(500)
        cap("Premier étage, fusion, reranking appris, lecture extractive", 3.5)
        top()

        # 7. Résultats
        tab("Performances")
        cap("Performances mesurées sur 3 061 questions de 16 articles jamais vus", 4)
        cap("Reranker +2,8 points de Hit@1 sur BM25 ; la RRF à poids égaux perd 22,9 points", 4.5)
        scroll(700, pause=60)
        cap("Chaque graphique indique sa source : table, fichier CSV et script producteur", 4.5)
        scroll(700, pause=60)
        pg.wait_for_timeout(1500)
        top()

        # 7b. Base de données
        tab("Base de données")
        cap("Onglet « Base de données » : d'où viennent les chiffres (SQLite, data/lumina.db)", 4)
        scroll(400)
        cap("Catalogue des tables : fichier source, script producteur, nombre de lignes", 4)
        scroll(500)
        cap("Explorateur de table et requêtes SQL en lecture seule", 4)
        move_click(pg.get_by_role("button", name="Exécuter la requête"))
        pg.wait_for_timeout(2500)
        scroll(300)
        pg.wait_for_timeout(1500)
        top()

        # 8. Données
        tab("propos")
        cap("À propos : données ouvertes, limites assumées, technologies", 4.5)
        scroll(300)
        pg.wait_for_timeout(1500)
        top()

        # 9. Mes documents
        top()
        move_click(pg.locator("label", has_text="Mes documents").first)
        pg.wait_for_timeout(2000)
        cap("Mes documents : fichiers ou texte collé, l'index est reconstruit à la volée", 3.5)
        ta = pg.locator("textarea").first
        move_click(ta)
        docs = ("Casablanca is the largest city of Morocco and its economic capital. The Hassan II Mosque, completed in 1993, "
                "stands on the Atlantic coast and has the tallest minaret in the world.\n\n"
                "Rabat is the capital of Morocco. It is located on the Atlantic coast, at the mouth of the Bou Regreg river, "
                "and has been the seat of government since 1912.\n\n"
                "Marrakesh is a major tourist city in the west of Morocco, known for the Jemaa el-Fnaa square and the Koutoubia mosque.")
        pg.keyboard.insert_text(docs)
        pg.keyboard.press("Control+Enter")
        pg.wait_for_timeout(8000)
        qi = pg.get_by_placeholder("Posez votre question")
        move_click(qi)
        pg.keyboard.press("Control+A")
        pg.keyboard.type("Which city has the tallest minaret in the world?", delay=45)
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(3500)
        cap("Réponse sourcée sur les documents collés, avec export JSON et CSV", 4.5)
        pg.wait_for_timeout(1000)

        # 10. API
        cap("Même moteur exposé par une API REST (FastAPI)", 2.5)
        pg.goto(f"{API}/docs")
        pg.wait_for_timeout(2500)
        cap("Documentation interactive de l'API : routes /health et /search", 3.5)
        pg.goto(f"{API}/search?q=What%20dynasty%20came%20after%20the%20Yuan%3F&top=2")
        pg.wait_for_timeout(2000)
        cap("Réponse JSON : phrase retenue, passages classés et temps par étage", 4.5)
        cap("Code, données et instructions : dépôt Git, README et docker compose", 4)
        pg.close()
        ctx.close()
        b.close()
    webm = max(tmp.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    mp4 = OUT / "demo_plateforme_RAG.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("OK", mp4)


if __name__ == "__main__":
    main()
