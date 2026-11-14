import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data" / "squad_dev-v1.1.json"
GLOVE = ROOT / "data" / "glove-wiki-gigaword-100.gz"
RES = ROOT / "results"
FIG = RES / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
GREEN, GREY, DARK = "#009B7A", "#7A7F85", "#2B2F33"
ACCENT = "#E4572E"

def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GREY, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.prop_cycle": mpl.cycler(color=[GREEN, GREY, ACCENT, "#2E86AB", "#F2A541", "#6D4C9F"]),
    })
