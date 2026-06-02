from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def ouvrir_fichiers(chemins: list[Path], titre: str = "Fichiers generes", limite: int = 6) -> None:
    """Ouvre quelques fichiers lisibles pour rendre chaque etape visible en demo."""
    if os.environ.get("OUVRIR_RESULTATS", "1") == "0":
        return

    fichiers = [Path(chemin) for chemin in chemins if Path(chemin).exists()]
    if not fichiers:
        print("Aucun fichier a ouvrir pour cette etape.")
        return

    print(f"\n{titre} :")
    for chemin in fichiers[:limite]:
        print(f"- {chemin}")
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(chemin))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(chemin)])
            else:
                subprocess.Popen(["xdg-open", str(chemin)])
            time.sleep(0.2)
        except OSError as erreur:
            print(f"Ouverture impossible pour {chemin} : {erreur}")
