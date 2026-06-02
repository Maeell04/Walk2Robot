from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from marche_v2.caracteristiques import creer_fenetres
from marche_v2.config import DOSSIER_CARAC, DOSSIER_FEN, DOSSIER_PRET, DUREE_FEN, FREQ_CIBLE, PAS_FEN, creer_dossiers
from marche_v2.fichiers import ecrire_json, lister_csv
from marche_v2.ouverture import ouvrir_fichiers


def main() -> None:
    creer_dossiers()
    taille = int(DUREE_FEN * FREQ_CIBLE)
    pas = int(PAS_FEN * FREQ_CIBLE)
    blocs = []
    for chemin in lister_csv(DOSSIER_PRET):
        if chemin.name.startswith("resume_") or chemin.name.startswith("manifest_"):
            continue
        df = pd.read_csv(chemin)
        colonnes = [c for c in df.select_dtypes("number").columns if c not in ["phase", "echantillon", "temps", "pct_cycle_marche", "rfo", "lfo"]]
        blocs.append(creer_fenetres(df, colonnes, taille, pas))
    carac = pd.concat(blocs, ignore_index=True)
    noms = [c for c in carac.columns if c not in ["phase", "id_sujet", "id_essai", "debut", "fin"]]
    chemin_carac = DOSSIER_CARAC / "caracteristiques.csv"
    chemin_noms = DOSSIER_CARAC / "noms_caracteristiques.json"
    carac.to_csv(chemin_carac, index=False)
    ecrire_json(chemin_noms, {"noms": noms})
    np.save(DOSSIER_FEN / "fenetres_X.npy", carac[noms].to_numpy(dtype=float))
    np.save(DOSSIER_FEN / "fenetres_y.npy", carac["phase"].to_numpy(dtype=int))
    np.save(DOSSIER_FEN / "ids_sujets.npy", carac["id_sujet"].to_numpy())
    np.save(DOSSIER_FEN / "ids_essais.npy", carac["id_essai"].to_numpy())
    print(f"Caracteristiques extraites : {carac.shape[0]} fenetres, {len(noms)} variables.")
    ouvrir_fichiers([chemin_carac, chemin_noms], "Fichiers crees par l'etape 06")


if __name__ == "__main__":
    main()
