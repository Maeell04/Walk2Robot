from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ecrire_json(chemin: Path, donnees: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(donnees, indent=2, ensure_ascii=False), encoding="utf-8")


def lire_json(chemin: Path) -> dict:
    return json.loads(chemin.read_text(encoding="utf-8"))


def lister_csv(dossier: Path) -> list[Path]:
    return sorted(dossier.glob("*.csv"))


def lire_csv(chemin: Path) -> pd.DataFrame:
    return pd.read_csv(chemin)
