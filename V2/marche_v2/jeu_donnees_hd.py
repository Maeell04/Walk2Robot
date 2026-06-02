from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import DOSSIER_HD, DOSSIER_MAT, DOSSIER_REEL
from .donnees_synth import CANAUX_EMG_HD


CHEMINS_MARCHE = [
    ("Level_Ground", "Walking", "Self_Selected_Speed"),
    ("Level_Ground", "Walking", "Slow_Speed"),
    ("Level_Ground", "Walking", "Fast_Speed"),
]

LIB_PHASE_REEL = {0: "appui_droit", 1: "balancement_droit"}


def id_participant(chemin: Path) -> str:
    return chemin.stem.upper()


def extraire_mat_participants() -> list[Path]:
    DOSSIER_MAT.mkdir(parents=True, exist_ok=True)
    chemins = []
    for zip_path in sorted(DOSSIER_HD.glob("P*.zip")):
        participant = id_participant(zip_path)
        cible = DOSSIER_MAT / participant / f"{participant}.mat"
        if cible.exists():
            chemins.append(cible)
            continue
        with zipfile.ZipFile(zip_path) as archive:
            membre = f"{participant}/{participant}.mat"
            cible.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(membre) as src, cible.open("wb") as dst:
                dst.write(src.read())
        chemins.append(cible)
    return chemins


def charger_mat(chemin: Path) -> dict[str, Any]:
    participant = chemin.stem.upper()
    donnees = loadmat(chemin, simplify_cells=True)
    return donnees[participant]


def lire_niveau(racine: dict[str, Any], cles: tuple[str, ...]) -> Any:
    obj: Any = racine
    for cle in cles:
        obj = obj[cle]
    return obj


def labels_depuis_evenement(nb: int, rfo: float, lfo: float | None = None) -> np.ndarray:
    vals = [v for v in [rfo, lfo] if v is not None and not np.isnan(v)]
    echelle = 200.0 if vals and max(vals) > 100 else 100.0
    frac = min(max(float(rfo) / echelle, 0.0), 1.0)
    pos = np.arange(nb) / nb
    labels = np.zeros(nb, dtype=int)
    labels[pos >= frac] = 1
    return labels


def essai_vers_dataframe(participant: str, activite: str, condition: str, idx: int, essai: dict[str, Any]) -> pd.DataFrame | None:
    if "RightLeg_EMG" not in essai or "RightLeg_IMU" not in essai:
        return None
    emg = np.asarray(essai["RightLeg_EMG"], dtype=float)
    imu = essai["RightLeg_IMU"]
    nb = emg.shape[0]
    phase = labels_depuis_evenement(nb, float(essai.get("RFO", 60)), float(essai.get("LFO", np.nan)))
    df = pd.DataFrame(
        {
            "id_sujet": participant,
            "id_essai": f"{activite}_{condition}_{idx:02d}",
            "activite": activite,
            "condition": condition,
            "echantillon": np.arange(nb),
            "temps": np.linspace(0, 1, nb, endpoint=False),
            "pct_cycle_marche": np.linspace(0, 100, nb, endpoint=False),
            "phase": phase,
            "nom_phase": [LIB_PHASE_REEL[int(x)] for x in phase],
            "rfo": float(essai.get("RFO", np.nan)),
            "lfo": float(essai.get("LFO", np.nan)),
            "acc_x": np.asarray(imu["AccX"], dtype=float),
            "acc_y": np.asarray(imu["AccY"], dtype=float),
            "acc_z": np.asarray(imu["AccZ"], dtype=float),
            "gyro_x": np.asarray(imu["GyroX"], dtype=float),
            "gyro_y": np.asarray(imu["GyroY"], dtype=float),
            "gyro_z": np.asarray(imu["GyroZ"], dtype=float),
        }
    )
    for i, canal in enumerate(CANAUX_EMG_HD):
        if i < emg.shape[1]:
            df[canal] = emg[:, i]
    return df


def exporter_essais_reels(max_participants: int | None = None) -> pd.DataFrame:
    DOSSIER_REEL.mkdir(parents=True, exist_ok=True)
    chemins = extraire_mat_participants()
    if max_participants is not None:
        chemins = chemins[:max_participants]
    lignes = []
    for chemin in chemins:
        participant = chemin.stem.upper()
        donnees = charger_mat(chemin)
        pied_droit = donnees["RightFoot_GaitCycle_Data"]
        for cle1, activite, condition in CHEMINS_MARCHE:
            try:
                essais = lire_niveau(pied_droit, (cle1, activite, condition))
            except KeyError:
                continue
            if isinstance(essais, dict):
                essais = [essais]
            for idx, essai in enumerate(essais, start=1):
                df = essai_vers_dataframe(participant, activite, condition, idx, essai)
                if df is None:
                    continue
                sortie = DOSSIER_REEL / f"{participant}_{activite}_{condition}_{idx:02d}.csv"
                df.to_csv(sortie, index=False)
                lignes.append(
                    {
                        "id_sujet": participant,
                        "id_essai": df["id_essai"].iloc[0],
                        "fichier": str(sortie.relative_to(DOSSIER_REEL.parents[1])),
                        "echantillons": int(len(df)),
                        "duree_cycle_normalisee": 1.0,
                        "canaux_emg": len([c for c in df.columns if c.startswith("emg")]),
                        "canaux_imu": 6,
                        "evenements": "RFO,LFO",
                        "condition": condition,
                        "source": "reelle",
                    }
                )
    manifest = pd.DataFrame(lignes)
    manifest.to_csv(DOSSIER_REEL / "manifest_donnees_reelles.csv", index=False)
    return manifest
