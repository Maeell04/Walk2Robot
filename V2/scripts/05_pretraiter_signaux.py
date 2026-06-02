from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marche_v2.config import DOSSIER_PRET, DOSSIER_SYNC, creer_dossiers
from marche_v2.fichiers import lister_csv
from marche_v2.ouverture import ouvrir_fichiers


def main() -> None:
    creer_dossiers()
    lignes = []
    for chemin in lister_csv(DOSSIER_SYNC):
        if chemin.name.startswith("rapport_") or chemin.name.startswith("canaux_") or chemin.name.startswith("manifest_"):
            continue
        df = pd.read_csv(chemin)
        num = [c for c in df.select_dtypes("number").columns if c not in ["phase", "echantillon"]]
        for col in num:
            moy = df[col].mean()
            ecart = df[col].std()
            if ecart == 0 or pd.isna(ecart):
                ecart = 1.0
            df[col] = (df[col] - moy) / ecart
        sortie = DOSSIER_PRET / chemin.name.replace("_sync.csv", "_pretraite.csv")
        df.to_csv(sortie, index=False)
        lignes.append({"fichier": sortie.name, "colonnes_normalisees": len(num), "echantillons": len(df)})
    chemin_resume = DOSSIER_PRET / "resume_pretraitement.csv"
    pd.DataFrame(lignes).to_csv(chemin_resume, index=False)
    print(f"Pretraitement termine : {len(lignes)} fichiers.")
    ouvrir_fichiers([chemin_resume, *sorted(DOSSIER_PRET.glob("*_pretraite.csv"))[:2]], "Fichiers crees par l'etape 05")


if __name__ == "__main__":
    main()
