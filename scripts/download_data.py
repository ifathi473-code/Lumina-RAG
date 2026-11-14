"""Télécharge les données ouvertes (SQuAD v1.1 dev, GloVe 100 d) et vérifie leur empreinte SHA-256.
Sources et licences : voir data/DATA.md."""
import hashlib
import sys
import time
import urllib.request
from common import DATA, GLOVE

FILES = {
    DATA: ("https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json",
           "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"),
    GLOVE: ("https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz",
            "4934d4708bf3d65f28fc984dbbfa18af9cbd2505d166a7258a1a957808559b43"),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, tries=3):
    for k in range(1, tries + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # réseau instable ou bloqué
            print(f"  essai {k}/{tries} échoué : {e}")
            time.sleep(2 * k)
    sys.exit(f"Impossible de télécharger {url}.\nTéléchargez le fichier à la main et placez-le dans data/ (voir data/DATA.md).")


if __name__ == "__main__":
    DATA.parent.mkdir(exist_ok=True)
    for path, (url, digest) in FILES.items():
        if not path.exists():
            print("téléchargement", url)
            fetch(url, path)
        ok = sha256(path) == digest
        print(("OK    " if ok else "ATTENTION empreinte différente : ") + path.name)
        if not ok:
            print("  Le fichier a changé en amont ou est corrompu ; les résultats peuvent différer légèrement.")
    print("Données prêtes.")
