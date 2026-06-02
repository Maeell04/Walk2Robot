from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from marche_v2.config import DOSSIER_FEN, DOSSIER_MODELES, GRAINE, LIB_PHASE, creer_dossiers
from marche_v2.graphiques import graphe_comparaison_modeles, graphe_matrice_confusion
from marche_v2.modeles_simples import ClassifieurCentroide, NormaliseurMoyStd, matrice_confusion, metriques_classe, sauver_pickle
from marche_v2.ouverture import ouvrir_fichiers


def predire_centroide(x_appr: np.ndarray, y_appr: np.ndarray, x_eval: np.ndarray) -> tuple[ClassifieurCentroide, np.ndarray]:
    modele = ClassifieurCentroide().ajuster(x_appr, y_appr)
    return modele, modele.predire(x_eval)


def predire_sklearn(modele, x_appr: np.ndarray, y_appr: np.ndarray, x_eval: np.ndarray):
    modele.fit(x_appr, y_appr)
    return modele, modele.predict(x_eval)


def main() -> None:
    creer_dossiers()
    x = np.load(DOSSIER_FEN / "fenetres_X.npy")
    y = np.load(DOSSIER_FEN / "fenetres_y.npy")
    rng = np.random.default_rng(GRAINE)
    idx = rng.permutation(len(y))
    coupe = int(len(idx) * 0.8)
    apprentissage, evaluation = idx[:coupe], idx[coupe:]
    norm = NormaliseurMoyStd()
    x_appr = norm.ajuster_transformer(x[apprentissage])
    x_eval = norm.transformer(x[evaluation])
    y_appr = y[apprentissage]
    y_eval = y[evaluation]

    candidats = {
        "mlp": lambda: predire_sklearn(
            MLPClassifier(hidden_layer_sizes=(80, 40), max_iter=300, random_state=GRAINE),
            x_appr,
            y_appr,
            x_eval,
        ),
        "foret_aleatoire": lambda: predire_sklearn(
            RandomForestClassifier(n_estimators=120, random_state=GRAINE, class_weight="balanced"),
            x_appr,
            y_appr,
            x_eval,
        ),
        "naif_majoritaire": lambda: predire_sklearn(DummyClassifier(strategy="most_frequent"), x_appr, y_appr, x_eval),
    }

    lignes = []
    resultats = {}
    for nom, construire in candidats.items():
        modele, pred = construire()
        metr = metriques_classe(y_eval, pred)
        lignes.append({"modele": nom, **metr})
        resultats[nom] = {"modele": modele, "prediction": pred, "metriques": metr}

    df_metriques = pd.DataFrame(lignes).sort_values("f1_macro", ascending=False)
    meilleur_nom = str(df_metriques.iloc[0]["modele"])
    meilleur = resultats[meilleur_nom]
    chemin_metriques = DOSSIER_MODELES / "metriques_holdout.csv"
    chemin_matrice = DOSSIER_MODELES / "matrice_confusion.csv"
    chemin_graphe = DOSSIER_MODELES / "comparaison_modeles.png"
    chemin_graphe_matrice = DOSSIER_MODELES / "matrice_confusion.png"
    df_metriques.to_csv(chemin_metriques, index=False)
    classes = sorted(set(y.astype(int)))
    mat = matrice_confusion(y_eval, meilleur["prediction"], classes)
    pd.DataFrame(mat, index=[LIB_PHASE.get(c, str(c)) for c in classes], columns=[LIB_PHASE.get(c, str(c)) for c in classes]).to_csv(chemin_matrice)
    graphe_comparaison_modeles(df_metriques, chemin_graphe)
    graphe_matrice_confusion(mat, [LIB_PHASE.get(c, str(c)) for c in classes], chemin_graphe_matrice)
    sauver_pickle({"normaliseur": norm, "modele": meilleur["modele"], "nom_modele": meilleur_nom, "classes": classes}, DOSSIER_MODELES / "modele_phase_marche.pkl")
    print(
        f"Meilleur modele : {meilleur_nom}. "
        f"Exactitude={meilleur['metriques']['exactitude']:.3f}, "
        f"F1 macro={meilleur['metriques']['f1_macro']:.3f}"
    )
    ouvrir_fichiers([chemin_graphe, chemin_graphe_matrice, chemin_metriques, chemin_matrice], "Fichiers crees par l'etape 07")


if __name__ == "__main__":
    main()
