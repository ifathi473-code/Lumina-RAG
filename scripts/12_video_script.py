"""Régénère video/script_video_cameo.tex à partir des notes orales du PowerPoint."""
import json, re, collections
from pathlib import Path
from pptx import Presentation
R = Path(__file__).resolve().parents[1]
prs = Presentation(R / 'slides/presentation_RAG_ECC.pptx')
seq = json.load(open(R / 'slides/outline.json'))
def esc(t):
    t = t.replace('\\', '\\textbackslash{}')
    for a, b in [('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_'), ('$', '\\$')]: t = t.replace(a, b)
    return t
segs = []
for (n, who, title), sl in zip(seq, prs.slides):
    note = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ''
    note = re.sub(r'^\[[^\]]+\]\s*', '', note)
    sec = 140 if 'Démonstration' in title else round(len(note.split()) / 140 * 60)
    segs.append([n, who, title, sec, note])
blocks = []
for s in segs:
    if blocks and blocks[-1][0] == s[1]: blocks[-1][1].append(s)
    else: blocks.append([s[1], [s]])
mm = lambda x: f"{x//60}:{x%60:02d}"
t = 0; out = []; tab = []
for who, ss in blocks:
    start = t; body = []
    for n, w, title, sec, note in ss:
        body.append("\\seg{Diapo %d -- %s}{%s}{%s}{%s}" % (n, esc(title), mm(t), mm(sec), esc(note))); t += sec
    nums = f"{ss[0][0]}--{ss[-1][0]}" if len(ss) > 1 else str(ss[0][0])
    tab.append((who, nums, esc(', '.join(x[2] for x in ss))[:70], t - start))
    out.append("\\subsection*{%s -- diapositives %s}\n" % (esc(who), nums) + "\n".join(body))
table = "\\begin{center}\\small\n\\begin{tabularx}{\\textwidth}{llXr}\n\\toprule\n\\textbf{Intervenant} & \\textbf{Diapos} & \\textbf{Contenu} & \\textbf{Durée}\\\\\n\\midrule\n"
for who, nums, c, d in tab: table += f"{esc(who)} & {nums} & {c}\\ldots & {mm(d)}\\\\\n"
table += "\\midrule\n & & \\textbf{Total} & \\textbf{%s}\\\\\n\\bottomrule\n\\end{tabularx}\n\\end{center}\n" % mm(t)
p = R / 'video/script_video_cameo.tex'
old = p.read_text()
a = old.index("\\section*{1. Répartition"); b = old.index("\\section*{3. Enregistrement")
mid = "\\section*{1. Répartition et minutage}\n" + table + "Environ 140 mots par minute. Le texte vient des notes orales du PowerPoint ; reformulez avec vos mots mais gardez les chiffres exacts. Les durées sont des repères.\n\n\\section*{2. Texte à dire}\n" + "\n\n".join(out) + "\n\n"
new = old[:a] + mid + old[b:]
new = re.sub(r"durée cible [^\n(]*s ", "durée cible " + mm(t).replace(':', ' min ') + " s ", new, count=1)
p.write_text(new); print(mm(t))
