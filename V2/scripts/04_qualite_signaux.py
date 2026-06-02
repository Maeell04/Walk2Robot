from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from marche_v2.config import DOSSIER_SYNC, creer_dossiers
from marche_v2.fichiers import lister_csv
from marche_v2.ouverture import ouvrir_fichiers


def main() -> None:
    creer_dossiers()
    rapports, mauvais = [], []
    for chemin in lister_csv(DOSSIER_SYNC):
        if chemin.name.startswith("rapport_") or chemin.name.startswith("manifest_"):
            continue
        df = pd.read_csv(chemin)
        for col in df.select_dtypes("number").columns:
            x = df[col].to_numpy(dtype=float)
            ecart = float(np.nanstd(x))
            nan = int(np.isnan(x).sum())
            ok = nan == 0 and ecart > 1e-8
            rapports.append({"fichier": chemin.name, "canal": col, "ecart_type": ecart, "nan": nan, "ok": ok})
            if not ok:
                mauvais.append({"fichier": chemin.name, "canal": col, "raison": "nan ou signal constant"})
    chemin_rapport = DOSSIER_SYNC / "rapport_qualite.csv"
    chemin_mauvais = DOSSIER_SYNC / "canaux_mauvais.csv"
    pd.DataFrame(rapports).to_csv(chemin_rapport, index=False)
    pd.DataFrame(mauvais).to_csv(chemin_mauvais, index=False)
    print(f"Qualite controlee sur {len(rapports)} canaux.")
    ouvrir_fichiers([chemin_rapport, chemin_mauvais], "Fichiers crees par l'etape 04")


if __name__ == "__main__":
    main()
