from pathlib import Path


RACINE = Path(__file__).resolve().parents[1]
DOSSIER_BRUT = RACINE / "donnees_brutes"
DOSSIER_SYNTH = DOSSIER_BRUT / "donnees_synth"
DOSSIER_HD = DOSSIER_BRUT / "jeu_donnees_hd_emg_imu"
DOSSIER_REEL = DOSSIER_BRUT / "donnees_reelles"
DOSSIER_MAT = DOSSIER_HD / "mats_extraits"
DOSSIER_TRAITE = RACINE / "donnees_traitees"
DOSSIER_SYNC = DOSSIER_TRAITE / "synchronisees"
DOSSIER_PRET = DOSSIER_TRAITE / "pretraitees"
DOSSIER_CARAC = DOSSIER_TRAITE / "caracteristiques"
DOSSIER_FEN = DOSSIER_TRAITE / "fenetres"
DOSSIER_MODELES = RACINE / "modeles"
DOSSIER_RAPPORTS = RACINE.parent / "doc" / "V2" / "rapports"
DOSSIER_FIG = DOSSIER_RAPPORTS / "figures"

FREQ_CIBLE = 100
DUREE_FEN = 0.25
PAS_FEN = 0.10
GRAINE = 42

LIB_PHASE = {
    0: "appui_droit",
    1: "balancement_droit",
    2: "appui_gauche",
    3: "balancement_gauche",
}

JEU_DONNEES = {
    "nom": "High-density EMG, IMU, Kinetic, and Kinematic Open-Source Dataset",
    "url_figshare": "https://figshare.com/articles/dataset/High-density_EMG_IMU_Kinetic_and_Kinematic_Open-Source_Dataset/22227337",
    "doi": "https://doi.org/10.6084/m9.figshare.22227337.v2",
    "article": "https://doi.org/10.1038/s41597-023-02679-x",
    "licence": "CC BY 4.0",
}


def creer_dossiers() -> None:
    for dossier in [
        DOSSIER_BRUT,
        DOSSIER_SYNTH,
        DOSSIER_REEL,
        DOSSIER_MAT,
        DOSSIER_SYNC,
        DOSSIER_PRET,
        DOSSIER_CARAC,
        DOSSIER_FEN,
        DOSSIER_MODELES,
        DOSSIER_RAPPORTS,
        DOSSIER_FIG,
    ]:
        dossier.mkdir(parents=True, exist_ok=True)
