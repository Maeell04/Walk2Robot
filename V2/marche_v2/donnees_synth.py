from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DOSSIER_SYNTH, FREQ_CIBLE, GRAINE, LIB_PHASE


CANAUX_EMG = ["emg_tibial", "emg_gastro", "emg_quadri", "emg_ischio"]
CANAUX_EMG_HD = [f"emg_hd_{i:02d}" for i in range(1, 33)]
CANAUX_IMU = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CANAUX_MVT = ["angle_hanche", "angle_genou", "angle_cheville", "force_sol"]


def phase_depuis_cycle(cycle: np.ndarray) -> np.ndarray:
    phase = np.zeros_like(cycle, dtype=int)
    phase[(cycle >= 0.30) & (cycle < 0.50)] = 1
    phase[(cycle >= 0.50) & (cycle < 0.80)] = 2
    phase[cycle >= 0.80] = 3
    return phase


def generer_essai(id_sujet: int, id_essai: int, duree_s: float = 12.0) -> pd.DataFrame:
    rng = np.random.default_rng(GRAINE + id_sujet * 100 + id_essai)
    nb = int(duree_s * FREQ_CIBLE)
    temps = np.arange(nb) / FREQ_CIBLE
    freq_pas = rng.normal(1.05, 0.08)
    cycle = (temps * freq_pas + rng.uniform(0, 1)) % 1.0
    phase = phase_depuis_cycle(cycle)

    angle_hanche = 22 * np.sin(2 * np.pi * cycle) + rng.normal(0, 2.0, nb)
    angle_genou = 42 * np.maximum(0, np.sin(2 * np.pi * (cycle - 0.12))) + rng.normal(0, 2.5, nb)
    angle_cheville = 12 * np.sin(2 * np.pi * (cycle + 0.18)) + rng.normal(0, 1.5, nb)
    force_sol = 0.65 + 0.35 * (phase % 2 == 0).astype(float) + rng.normal(0, 0.04, nb)

    df = pd.DataFrame(
        {
            "id_sujet": f"S{id_sujet:02d}",
            "id_essai": f"essai_{id_essai:02d}",
            "temps": temps,
            "echantillon": np.arange(nb),
            "pct_cycle_marche": cycle * 100,
            "phase": phase,
            "nom_phase": [LIB_PHASE[int(x)] for x in phase],
            "angle_hanche": angle_hanche,
            "angle_genou": angle_genou,
            "angle_cheville": angle_cheville,
            "force_sol": force_sol,
            "acc_x": np.gradient(angle_hanche) + rng.normal(0, 0.08, nb),
            "acc_y": np.gradient(angle_genou) + rng.normal(0, 0.08, nb),
            "acc_z": 9.81 + 0.2 * np.sin(4 * np.pi * cycle) + rng.normal(0, 0.05, nb),
            "gyro_x": np.gradient(angle_hanche) * FREQ_CIBLE,
            "gyro_y": np.gradient(angle_genou) * FREQ_CIBLE,
            "gyro_z": np.gradient(angle_cheville) * FREQ_CIBLE,
        }
    )

    enveloppes = {
        "emg_tibial": np.clip(0.2 + 0.8 * (phase == 1) + rng.normal(0, 0.08, nb), 0, None),
        "emg_gastro": np.clip(0.2 + 0.8 * (phase == 0) + rng.normal(0, 0.08, nb), 0, None),
        "emg_quadri": np.clip(0.2 + 0.7 * (phase == 2) + rng.normal(0, 0.08, nb), 0, None),
        "emg_ischio": np.clip(0.2 + 0.7 * (phase == 3) + rng.normal(0, 0.08, nb), 0, None),
    }
    for nom, val in enveloppes.items():
        df[nom] = val
    for i, nom in enumerate(CANAUX_EMG_HD):
        base = enveloppes[CANAUX_EMG[i % len(CANAUX_EMG)]]
        df[nom] = np.clip(base + rng.normal(0, 0.05, nb), 0, None)
    return df


def creer_jeu_donnees_synth(nb_sujets: int = 5, nb_essais: int = 3) -> list[dict]:
    DOSSIER_SYNTH.mkdir(parents=True, exist_ok=True)
    lignes = []
    for sujet in range(1, nb_sujets + 1):
        for essai in range(1, nb_essais + 1):
            df = generer_essai(sujet, essai)
            chemin = DOSSIER_SYNTH / f"sujet_{sujet:02d}_essai_{essai:02d}.csv"
            df.to_csv(chemin, index=False)
            lignes.append(
                {
                    "id_sujet": f"S{sujet:02d}",
                    "id_essai": f"essai_{essai:02d}",
                    "fichier": str(chemin.relative_to(DOSSIER_SYNTH.parents[1])),
                    "echantillons": int(len(df)),
                    "freq_hz": FREQ_CIBLE,
                    "source": "synthetique",
                }
            )
    return lignes
