from pathlib import Path

import numpy as np


RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DONNEES = RACINE / "donnees_brutes" / "donnees_amp"
FICHIERS = [
    "ergocub_walk.npy",
    "ergocub_walk_left0.npy",
    "ergocub_walk_right2.npy",
    "ergocub_stand_still.npy",
]


def charger_fichier(nom_fichier: str) -> dict:
    chemin = DOSSIER_DONNEES / nom_fichier
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    return np.load(chemin, allow_pickle=True).item()


def afficher_resume(nom_fichier: str, donnees: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Fichier : {nom_fichier}")
    print("=" * 70)
    print(f"Cles disponibles : {list(donnees.keys())}")
    for cle, val in donnees.items():
        if isinstance(val, np.ndarray):
            print(f"- {cle} : tableau {val.shape}, type {val.dtype}")
        elif isinstance(val, list):
            print(f"- {cle} : liste de {len(val)} elements")
        else:
            print(f"- {cle} : {val}")


def main() -> None:
    print("Lecture du dataset V1 ergoCub")
    for nom in FICHIERS:
        donnees = charger_fichier(nom)
        afficher_resume(nom, donnees)


if __name__ == "__main__":
    main()
