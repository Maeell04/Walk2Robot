from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ouverture import ouvrir_fichiers


RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DONNEES = RACINE / "donnees_traitees"
DOSSIER_MODELE = RACINE / "resultats_modele"
DOSSIER_SORTIE = RACINE / "resultats_robotique"
DOSSIER_SORTIE.mkdir(exist_ok=True)

ARTICULATIONS = [
    "hanche_droite",
    "genou_droit",
    "cheville_droite",
    "hanche_gauche",
    "genou_gauche",
    "cheville_gauche",
]


def charger_modele_et_sequences():
    modele = joblib.load(DOSSIER_MODELE / "modele_mlp_posture.joblib")
    x = np.load(DOSSIER_DONNEES / "sequences_X.npy")
    return modele, x.reshape(x.shape[0], x.shape[1] * x.shape[2])


def convertir_commandes(pred: np.ndarray) -> pd.DataFrame:
    cmd = pd.DataFrame(pred, columns=ARTICULATIONS)
    cmd["robot_epaule_avant_droite"] = cmd["hanche_droite"]
    cmd["robot_coude_avant_droit"] = cmd["genou_droit"]
    cmd["robot_poignet_avant_droit"] = cmd["cheville_droite"]
    cmd["robot_epaule_avant_gauche"] = cmd["hanche_gauche"]
    cmd["robot_coude_avant_gauche"] = cmd["genou_gauche"]
    cmd["robot_poignet_avant_gauche"] = cmd["cheville_gauche"]
    cmd["temps"] = np.arange(len(cmd)) / 100.0
    return cmd


def tracer_commandes(cmd: pd.DataFrame) -> None:
    cols = [c for c in cmd.columns if c.startswith("robot_")]
    plt.figure(figsize=(11, 6))
    for col in cols[:6]:
        plt.plot(cmd["temps"][:300], cmd[col][:300], label=col)
    plt.title("Commandes articulaires predites pour le robot")
    plt.xlabel("Temps (s)")
    plt.ylabel("Commande normalisee")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / "commandes_articulaires_predites.png")
    plt.close()


def ecrire_messages_ros(cmd: pd.DataFrame) -> None:
    lignes = []
    cols = [c for c in cmd.columns if c.startswith("robot_")]
    for i, ligne in cmd.head(20).iterrows():
        valeurs = {col: round(float(ligne[col]), 4) for col in cols}
        lignes.append(f"message_{i:03d}: {valeurs}")
    (DOSSIER_SORTIE / "messages_ros2_simules.txt").write_text("\n".join(lignes), encoding="utf-8")


def main() -> None:
    print("\nIntegration robotique V1")
    modele, x = charger_modele_et_sequences()
    pred = modele.predict(x)
    cmd = convertir_commandes(pred)
    chemin_cmd = DOSSIER_SORTIE / "commandes_articulaires_predites.csv"
    chemin_img = DOSSIER_SORTIE / "commandes_articulaires_predites.png"
    chemin_msg = DOSSIER_SORTIE / "messages_ros2_simules.txt"
    chemin_resume = DOSSIER_SORTIE / "resume_integration_robotique.txt"
    cmd.to_csv(chemin_cmd, index=False)
    tracer_commandes(cmd)
    ecrire_messages_ros(cmd)
    chemin_resume.write_text(
        "La V1 transforme les postures predites en commandes articulaires simplifiees pour une future integration robotique.\n",
        encoding="utf-8",
    )
    print(f"Commandes generees : {len(cmd)} lignes")
    ouvrir_fichiers([chemin_cmd, chemin_img, chemin_msg, chemin_resume], "Fichiers crees par l'integration robotique V1")


if __name__ == "__main__":
    main()
