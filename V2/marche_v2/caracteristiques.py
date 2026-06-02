from __future__ import annotations

import numpy as np
import pandas as pd


def label_majoritaire(valeurs: np.ndarray) -> int:
    valeurs = valeurs.astype(int)
    return int(np.bincount(valeurs).argmax())


def calculer_carac(fenetre: pd.DataFrame, colonnes: list[str]) -> dict[str, float]:
    carac: dict[str, float] = {}
    for col in colonnes:
        x = fenetre[col].to_numpy(dtype=float)
        carac[f"{col}_moy"] = float(np.mean(x))
        carac[f"{col}_std"] = float(np.std(x))
        carac[f"{col}_min"] = float(np.min(x))
        carac[f"{col}_max"] = float(np.max(x))
        carac[f"{col}_energie"] = float(np.mean(x * x))
        carac[f"{col}_rms"] = float(np.sqrt(np.mean(x * x)))
    return carac


def creer_fenetres(df: pd.DataFrame, colonnes: list[str], taille: int, pas: int) -> pd.DataFrame:
    lignes = []
    for debut in range(0, len(df) - taille + 1, pas):
        fin = debut + taille
        fen = df.iloc[debut:fin]
        ligne = calculer_carac(fen, colonnes)
        ligne["phase"] = label_majoritaire(fen["phase"].to_numpy())
        ligne["id_sujet"] = fen["id_sujet"].iloc[0]
        ligne["id_essai"] = fen["id_essai"].iloc[0]
        ligne["debut"] = int(debut)
        ligne["fin"] = int(fin)
        lignes.append(ligne)
    return pd.DataFrame(lignes)
