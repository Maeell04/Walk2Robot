from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def graphe_comparaison_modeles(df: pd.DataFrame, chemin: Path) -> None:
    noms = df["modele"].to_list()
    x = np.arange(len(noms))
    largeur = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - largeur, df["exactitude"], width=largeur, label="Exactitude")
    ax.bar(x, df["exactitude_equilibree"], width=largeur, label="Exactitude equilibree")
    ax.bar(x + largeur, df["f1_macro"], width=largeur, label="F1 macro")
    ax.set_title("Comparaison des modeles")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=160)
    plt.close(fig)


def graphe_matrice_confusion(mat: np.ndarray, labels: list[str], chemin: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(mat, cmap="Blues")
    ax.set_title("Matrice de confusion du meilleur modele")
    ax.set_xlabel("Phase predite")
    ax.set_ylabel("Phase reelle")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="black")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=160)
    plt.close(fig)


def graphe_scores_sujets(df: pd.DataFrame, chemin: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["exactitude"], width=0.36, label="Exactitude")
    ax.bar(x + 0.18, df["f1_macro"], width=0.36, label="F1 macro")
    ax.set_title("Evaluation leave-one-subject-out")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(df["id_sujet_test"], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=160)
    plt.close(fig)
