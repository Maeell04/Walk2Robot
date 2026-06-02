from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marche_v2.config import DOSSIER_REEL, DOSSIER_SYNC, DOSSIER_SYNTH, FREQ_CIBLE, creer_dossiers
from marche_v2.fichiers import lister_csv
from marche_v2.ouverture import ouvrir_fichiers


def lister_essais(dossier: Path) -> list[Path]:
    return [chemin for chemin in lister_csv(dossier) if not chemin.name.startswith("manifest_")]


def main() -> None:
    creer_dossiers()
    source = DOSSIER_REEL if lister_essais(DOSSIER_REEL) else DOSSIER_SYNTH
    lignes = []
    for chemin in lister_essais(source):
        df = pd.read_csv(chemin).sort_values("temps")
        temps = df["temps"].to_numpy()
        debut, fin = float(temps.min()), float(temps.max())
        grille = pd.DataFrame({"temps": pd.Series([debut + i / FREQ_CIBLE for i in range(int((fin - debut) * FREQ_CIBLE) + 1)])})
        num = df.select_dtypes("number").columns.tolist()
        cat = [c for c in df.columns if c not in num and c != "temps"]
        sync = grille.copy()
        for col in num:
            sync[col] = pd.Series(pd.Series(df[col].to_numpy(), index=temps).reindex(grille["temps"], method=None).interpolate().bfill().ffill().to_numpy())
        for col in cat:
            sync[col] = df[col].iloc[0]
        if "phase" in sync:
            sync["phase"] = sync["phase"].round().astype(int)
        sortie = DOSSIER_SYNC / f"{chemin.stem}_sync.csv"
        sync.to_csv(sortie, index=False)
        lignes.append({"fichier": sortie.name, "echantillons": len(sync), "freq_hz": FREQ_CIBLE})
    chemin_rapport = DOSSIER_SYNC / "rapport_sync.csv"
    pd.DataFrame(lignes).to_csv(chemin_rapport, index=False)
    print(f"Synchronisation terminee : {len(lignes)} fichiers.")
    ouvrir_fichiers([chemin_rapport, *sorted(DOSSIER_SYNC.glob("*_sync.csv"))[:2]], "Fichiers crees par l'etape 03")


if __name__ == "__main__":
    main()
