from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ouverture import ouvrir_fichiers


RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DONNEES = RACINE / "donnees_brutes" / "donnees_amp"
DOSSIER_SORTIE = RACINE / "donnees_traitees"
DOSSIER_SORTIE.mkdir(exist_ok=True)
FICHIER_PRINCIPAL = "ergocub_walk_left0.npy"

ARTICULATIONS = [
    "r_hip_pitch",
    "r_knee",
    "r_ankle_pitch",
    "l_hip_pitch",
    "l_knee",
    "l_ankle_pitch",
]


def charger_dataset(nom_fichier: str) -> dict:
    chemin = DOSSIER_DONNEES / nom_fichier
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    return np.load(chemin, allow_pickle=True).item()


def verifier_nan(nom: str, tab: np.ndarray) -> int:
    nb = int(np.isnan(np.asarray(tab, dtype=float)).sum())
    print(f"NaN dans {nom} : {nb}")
    return nb


def verifier_qualite(donnees: dict) -> None:
    print("\nVerification de la qualite des donnees")
    for cle in ["joint_positions", "root_position", "root_quaternion", "l_sole", "r_sole"]:
        verifier_nan(cle, donnees[cle])


def normaliser(tab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tab = np.asarray(tab, dtype=float)
    moy = tab.mean(axis=0)
    ecart = tab.std(axis=0)
    ecart[ecart == 0] = 1
    return (tab - moy) / ecart, moy, ecart


def calculer_vitesse(tab: np.ndarray, fps: float) -> np.ndarray:
    tab = np.asarray(tab, dtype=float)
    vit = np.diff(tab, axis=0) * fps
    return np.vstack([np.zeros((1, tab.shape[1])), vit])


def extraire_articulations(positions: np.ndarray, liste_art: list[str]) -> tuple[np.ndarray, list[int]]:
    indices = []
    for art in ARTICULATIONS:
        if art not in liste_art:
            raise ValueError(f"Articulation introuvable : {art}")
        indices.append(liste_art.index(art))
    return positions[:, indices], indices


def creer_table_carac(donnees: dict) -> tuple[pd.DataFrame, dict]:
    fps = donnees["fps"]
    liste_art = donnees["joints_list"]
    pos_art = np.asarray(donnees["joint_positions"], dtype=float)
    pos_base = np.asarray(donnees["root_position"], dtype=float)
    pied_g = np.asarray(donnees["l_sole"], dtype=float)
    pied_d = np.asarray(donnees["r_sole"], dtype=float)
    nb = pos_art.shape[0]
    temps = np.arange(nb) / fps

    pos_sel, _ = extraire_articulations(pos_art, liste_art)
    vit_sel = calculer_vitesse(pos_sel, fps)
    vit_base = calculer_vitesse(pos_base, fps)
    pos_norm, moy_art, ecart_art = normaliser(pos_sel)
    base_norm, moy_base, ecart_base = normaliser(pos_base)

    df = pd.DataFrame({"temps": temps})
    for i, art in enumerate(ARTICULATIONS):
        df[f"{art}_angle"] = pos_sel[:, i]
        df[f"{art}_angle_norm"] = pos_norm[:, i]
        df[f"{art}_vitesse"] = vit_sel[:, i]
    df["base_x"], df["base_y"], df["base_z"] = pos_base[:, 0], pos_base[:, 1], pos_base[:, 2]
    df["base_x_norm"], df["base_y_norm"], df["base_z_norm"] = base_norm[:, 0], base_norm[:, 1], base_norm[:, 2]
    df["base_vx"], df["base_vy"], df["base_vz"] = vit_base[:, 0], vit_base[:, 1], vit_base[:, 2]
    df["hauteur_pied_gauche"] = pied_g[:, 2]
    df["hauteur_pied_droit"] = pied_d[:, 2]
    df["ecart_hauteur_pieds"] = df["hauteur_pied_gauche"] - df["hauteur_pied_droit"]
    df["pied_gauche_x"], df["pied_gauche_y"], df["pied_gauche_z"] = pied_g[:, 0], pied_g[:, 1], pied_g[:, 2]
    df["pied_droit_x"], df["pied_droit_y"], df["pied_droit_z"] = pied_d[:, 0], pied_d[:, 1], pied_d[:, 2]

    params = {
        "moy_articulations": moy_art,
        "ecart_articulations": ecart_art,
        "moy_base": moy_base,
        "ecart_base": ecart_base,
        "articulations": ARTICULATIONS,
        "fps": fps,
    }
    return df, params


def creer_sequences(df: pd.DataFrame, entrees: list[str], cibles: list[str], taille: int = 10) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    val_x = df[entrees].to_numpy()
    val_y = df[cibles].to_numpy()
    for i in range(taille, len(df)):
        x.append(val_x[i - taille:i])
        y.append(val_y[i])
    return np.asarray(x), np.asarray(y)


def tracer(df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(df["temps"], df["r_knee_angle"], label="brut")
    plt.plot(df["temps"], df["r_knee_angle_norm"], label="normalise")
    plt.title("Genou droit : angle brut et normalise")
    plt.xlabel("Temps (s)")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)
    plt.savefig(DOSSIER_SORTIE / "controle_genou_droit.png")
    plt.close()


def main() -> None:
    print("\nPretraitement du dataset ergoCub")
    donnees = charger_dataset(FICHIER_PRINCIPAL)
    print(f"Fichier charge : {FICHIER_PRINCIPAL}")
    print(f"FPS : {donnees['fps']}")
    print(f"Duree : {donnees['duration']}")
    print(f"Frames : {len(donnees['joint_positions'])}")
    verifier_qualite(donnees)

    df, params = creer_table_carac(donnees)
    print(f"Table creee : {df.shape}")
    chemin_carac = DOSSIER_SORTIE / "caracteristiques_ergocub.csv"
    chemin_params = DOSSIER_SORTIE / "parametres_normalisation.npy"
    df.to_csv(chemin_carac, index=False)
    np.save(chemin_params, params, allow_pickle=True)

    colonnes = [f"{art}_angle_norm" for art in ARTICULATIONS]
    x, y = creer_sequences(df, colonnes, colonnes, taille=10)
    np.save(DOSSIER_SORTIE / "sequences_X.npy", x)
    np.save(DOSSIER_SORTIE / "cibles_y.npy", y)
    tracer(df)
    print(f"Sequences X : {x.shape}")
    print(f"Cibles y : {y.shape}")
    print("Pretraitement termine.")
    ouvrir_fichiers([chemin_carac, DOSSIER_SORTIE / "controle_genou_droit.png"], "Fichiers crees par le pretraitement V1")


if __name__ == "__main__":
    main()
