from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from ouverture import ouvrir_fichiers


RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DONNEES = RACINE / "donnees_traitees"
DOSSIER_SORTIE = RACINE / "resultats_modele"
DOSSIER_SORTIE.mkdir(exist_ok=True)

ARTICULATIONS = [
    "hanche_droite",
    "genou_droit",
    "cheville_droite",
    "hanche_gauche",
    "genou_gauche",
    "cheville_gauche",
]


def charger_sequences() -> tuple[np.ndarray, np.ndarray]:
    x = np.load(DOSSIER_DONNEES / "sequences_X.npy")
    y = np.load(DOSSIER_DONNEES / "cibles_y.npy")
    return x, y


def aplatir_sequences(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2])


def entrainer(x_train: np.ndarray, y_train: np.ndarray) -> MLPRegressor:
    modele = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", max_iter=500, random_state=42)
    modele.fit(x_train, y_train)
    return modele


def evaluer(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }


def tracer_erreurs(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    mae_art = np.mean(np.abs(y_test - y_pred), axis=0)
    plt.figure(figsize=(9, 4))
    plt.bar(ARTICULATIONS, mae_art)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Erreur absolue moyenne")
    plt.title("Erreur par articulation")
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / "erreur_par_articulation.png")
    plt.close()


def tracer_prediction(y_test: np.ndarray, y_pred: np.ndarray, idx: int, nom: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(y_test[:200, idx], label="reel")
    plt.plot(y_pred[:200, idx], label="predit")
    plt.title(f"Prediction : {nom}")
    plt.xlabel("Echantillon")
    plt.ylabel("Angle normalise")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / f"prediction_{nom}.png")
    plt.close()


def main() -> None:
    print("\nEntrainement du modele V1")
    x, y = charger_sequences()
    x_flat = aplatir_sequences(x)
    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, test_size=0.2, random_state=42, shuffle=False)
    modele = entrainer(x_train, y_train)
    y_pred = modele.predict(x_test)
    met = evaluer(y_test, y_pred)
    chemin_metriques = DOSSIER_SORTIE / "metriques.txt"
    chemin_erreurs = DOSSIER_SORTIE / "erreur_par_articulation.png"
    chemin_pred_d = DOSSIER_SORTIE / "prediction_genou_droit.png"
    chemin_pred_g = DOSSIER_SORTIE / "prediction_genou_gauche.png"
    joblib.dump(modele, DOSSIER_SORTIE / "modele_mlp_posture.joblib")
    np.save(DOSSIER_SORTIE / "cibles_test.npy", y_test)
    np.save(DOSSIER_SORTIE / "predictions.npy", y_pred)
    chemin_metriques.write_text(
        "\n".join([f"MAE : {met['mae']:.6f}", f"RMSE : {met['rmse']:.6f}", f"R2 : {met['r2']:.6f}"]),
        encoding="utf-8",
    )
    tracer_erreurs(y_test, y_pred)
    tracer_prediction(y_test, y_pred, 1, "genou_droit")
    tracer_prediction(y_test, y_pred, 4, "genou_gauche")
    print(f"MAE : {met['mae']:.6f}")
    print(f"RMSE : {met['rmse']:.6f}")
    print(f"R2 : {met['r2']:.6f}")
    print("Modele sauvegarde.")
    ouvrir_fichiers([chemin_metriques, chemin_erreurs, chemin_pred_d, chemin_pred_g], "Fichiers crees par l'entrainement V1")


if __name__ == "__main__":
    main()
