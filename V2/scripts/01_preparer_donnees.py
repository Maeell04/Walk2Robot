from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from marche_v2.config import JEU_DONNEES, DOSSIER_BRUT, DOSSIER_HD, creer_dossiers
from marche_v2.jeu_donnees_hd import exporter_essais_reels
from marche_v2.donnees_synth import creer_jeu_donnees_synth
from marche_v2.ouverture import ouvrir_fichiers


def main() -> None:
    creer_dossiers()
    note = DOSSIER_BRUT / "source_jeu_donnees.md"
    note.write_text(
        "\n".join(
            [
                "# Jeu de donnees V2 recommande",
                "",
                f"Nom : {JEU_DONNEES['nom']}",
                f"Figshare : {JEU_DONNEES['url_figshare']}",
                f"DOI : {JEU_DONNEES['doi']}",
                f"Article : {JEU_DONNEES['article']}",
                f"Licence : {JEU_DONNEES['licence']}",
                "",
                "Les archives P01.zip a P10.zip doivent etre placees dans donnees_brutes/jeu_donnees_hd_emg_imu.",
                "Si elles sont absentes, le pipeline genere un petit jeu synthetique pour tester l'execution.",
            ]
        ),
        encoding="utf-8",
    )

    if list(DOSSIER_HD.glob("P*.zip")):
        manifest = exporter_essais_reels()
        chemin_manifest = DOSSIER_BRUT / "donnees_reelles" / "manifest_donnees_reelles.csv"
        exemples = sorted((DOSSIER_BRUT / "donnees_reelles").glob("*.csv"))[:2]
        print(f"Essais reels extraits : {len(manifest)}")
        print(f"Manifest : {chemin_manifest}")
        ouvrir_fichiers([note, chemin_manifest, *exemples], "Fichiers crees par l'etape 01")
        return

    lignes = creer_jeu_donnees_synth()
    chemin_manifest = DOSSIER_BRUT / "manifest_donnees_synth.csv"
    pd.DataFrame(lignes).to_csv(chemin_manifest, index=False)
    print(f"Note source ecrite : {note}")
    print(f"Essais synthetiques generes : {len(lignes)}")
    ouvrir_fichiers([note, chemin_manifest, *sorted((DOSSIER_BRUT / "donnees_synth").glob("*.csv"))[:2]], "Fichiers crees par l'etape 01")


if __name__ == "__main__":
    main()
