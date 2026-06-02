from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from marche_v2.config import DOSSIER_FEN, DOSSIER_MODELES, creer_dossiers
from marche_v2.graphiques import graphe_scores_sujets
from marche_v2.modeles_simples import NormaliseurMoyStd, metriques_classe
from marche_v2.ouverture import ouvrir_fichiers


def main() -> None:
    creer_dossiers()
    x = np.load(DOSSIER_FEN / "fenetres_X.npy")
    y = np.load(DOSSIER_FEN / "fenetres_y.npy")
    sujets = np.load(DOSSIER_FEN / "ids_sujets.npy", allow_pickle=True)
    lignes = []
    for sujet in sorted(set(sujets)):
        evaluation = sujets == sujet
        apprentissage = ~evaluation
        norm = NormaliseurMoyStd()
        x_appr = norm.ajuster_transformer(x[apprentissage])
        x_eval = norm.transformer(x[evaluation])
        clf = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
        clf.fit(x_appr, y[apprentissage])
        pred = clf.predict(x_eval)
        metr = metriques_classe(y[evaluation], pred)
        metr["id_sujet_test"] = sujet
        metr["nb_fenetres"] = int(evaluation.sum())
        lignes.append(metr)
    chemin_metriques = DOSSIER_MODELES / "metriques_par_sujet.csv"
    chemin_graphe = DOSSIER_MODELES / "scores_par_sujet.png"
    df = pd.DataFrame(lignes)
    df.to_csv(chemin_metriques, index=False)
    graphe_scores_sujets(df, chemin_graphe)
    print(f"Evaluation par sujet terminee : {len(lignes)} sujets.")
    ouvrir_fichiers([chemin_graphe, chemin_metriques], "Fichiers crees par l'etape 08")


if __name__ == "__main__":
    main()
