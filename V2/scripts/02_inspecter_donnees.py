from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marche_v2.config import DOSSIER_REEL, DOSSIER_RAPPORTS, DOSSIER_SYNTH, creer_dossiers
from marche_v2.fichiers import lister_csv
from marche_v2.ouverture import ouvrir_fichiers


def lister_essais(dossier: Path) -> list[Path]:
    return [chemin for chemin in lister_csv(dossier) if not chemin.name.startswith("manifest_")]


def main() -> None:
    creer_dossiers()
    source = DOSSIER_REEL if lister_essais(DOSSIER_REEL) else DOSSIER_SYNTH
    lignes, sujets, modalites = [], [], []
    for chemin in lister_essais(source):
        df = pd.read_csv(chemin)
        if "temps" not in df.columns:
            print(f"Fichier ignore car ce n'est pas un essai temporel : {chemin.name}")
            continue
        lignes.append(
            {
                "fichier": chemin.name,
                "id_sujet": df["id_sujet"].iloc[0],
                "id_essai": df["id_essai"].iloc[0],
                "echantillons": len(df),
                "duree": float(df["temps"].max() - df["temps"].min()),
                "nb_colonnes": len(df.columns),
            }
        )
        sujets.append({"id_sujet": df["id_sujet"].iloc[0], "fichier": chemin.name})
        for prefixe in ["emg", "acc", "gyro", "angle", "force"]:
            modalites.append({"fichier": chemin.name, "modalite": prefixe, "nb_canaux": sum(c.startswith(prefixe) for c in df.columns)})
    chemin_resume = DOSSIER_RAPPORTS / "resume_jeu_donnees.csv"
    chemin_sujets = DOSSIER_RAPPORTS / "resume_sujets.csv"
    chemin_modalites = DOSSIER_RAPPORTS / "resume_modalites.csv"
    pd.DataFrame(lignes).to_csv(chemin_resume, index=False)
    pd.DataFrame(sujets).drop_duplicates().to_csv(chemin_sujets, index=False)
    pd.DataFrame(modalites).to_csv(chemin_modalites, index=False)
    print(f"Inspection terminee sur {len(lignes)} fichiers depuis {source}.")
    ouvrir_fichiers([chemin_resume, chemin_sujets, chemin_modalites], "Fichiers crees par l'etape 02")


if __name__ == "__main__":
    main()
