from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marche_v2.config import JEU_DONNEES, DOSSIER_CARAC, DOSSIER_MODELES, LIB_PHASE, creer_dossiers
from marche_v2.fichiers import ecrire_json, lire_json
from marche_v2.ouverture import ouvrir_fichiers


def main() -> None:
    creer_dossiers()
    noms = lire_json(DOSSIER_CARAC / "noms_caracteristiques.json")["noms"]
    met = pd.read_csv(DOSSIER_MODELES / "metriques_holdout.csv").iloc[0].to_dict()
    fiche = {
        "nom_modele": "classifieur_centroide_phase_marche",
        "objectif": "Predire la phase de marche a partir de fenetres EMG/IMU.",
        "jeu_donnees": JEU_DONNEES,
        "phases": LIB_PHASE,
        "nb_caracteristiques": len(noms),
        "metriques_holdout": met,
        "fichier_modele": "modele_phase_marche.pkl",
    }
    chemin_fiche = DOSSIER_MODELES / "fiche_modele.json"
    chemin_manifest = DOSSIER_MODELES / "manifest_export.json"
    ecrire_json(chemin_fiche, fiche)
    ecrire_json(
        chemin_manifest,
        {
            "fichiers": [
                "modele_phase_marche.pkl",
                "fiche_modele.json",
                "metriques_holdout.csv",
                "matrice_confusion.csv",
                "comparaison_modeles.png",
                "matrice_confusion.png",
                "scores_par_sujet.png",
            ]
        },
    )
    print("Export du modele termine.")
    ouvrir_fichiers([chemin_fiche, chemin_manifest], "Fichiers crees par l'etape 09")


if __name__ == "__main__":
    main()
