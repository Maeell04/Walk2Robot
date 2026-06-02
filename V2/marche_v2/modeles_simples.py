from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class NormaliseurMoyStd:
    def __init__(self) -> None:
        self.moy: np.ndarray | None = None
        self.ecart: np.ndarray | None = None

    def ajuster(self, x: np.ndarray) -> "NormaliseurMoyStd":
        self.moy = x.mean(axis=0)
        self.ecart = x.std(axis=0)
        self.ecart[self.ecart == 0] = 1.0
        return self

    def transformer(self, x: np.ndarray) -> np.ndarray:
        if self.moy is None or self.ecart is None:
            raise RuntimeError("Le normaliseur doit etre ajuste avant transformation.")
        return (x - self.moy) / self.ecart

    def ajuster_transformer(self, x: np.ndarray) -> np.ndarray:
        return self.ajuster(x).transformer(x)


class ClassifieurCentroide:
    def __init__(self) -> None:
        self.centres: dict[int, np.ndarray] = {}

    def ajuster(self, x: np.ndarray, y: np.ndarray) -> "ClassifieurCentroide":
        self.centres = {int(cl): x[y == cl].mean(axis=0) for cl in sorted(np.unique(y))}
        return self

    def predire(self, x: np.ndarray) -> np.ndarray:
        classes = np.array(sorted(self.centres))
        mat = np.vstack([self.centres[int(cl)] for cl in classes])
        dist = ((x[:, None, :] - mat[None, :, :]) ** 2).sum(axis=2)
        return classes[np.argmin(dist, axis=1)]


def sauver_pickle(objet: object, chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("wb") as flux:
        pickle.dump(objet, flux)


def charger_pickle(chemin: Path) -> object:
    with chemin.open("rb") as flux:
        return pickle.load(flux)


def matrice_confusion(y_vrai: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> np.ndarray:
    mat = np.zeros((len(classes), len(classes)), dtype=int)
    pos = {cl: i for i, cl in enumerate(classes)}
    for vrai, pred in zip(y_vrai, y_pred):
        mat[pos[int(vrai)], pos[int(pred)]] += 1
    return mat


def metriques_classe(y_vrai: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    classes = sorted(set(y_vrai.astype(int)) | set(y_pred.astype(int)))
    exact = float(np.mean(y_vrai == y_pred))
    f1 = []
    rappels = []
    for cl in classes:
        vp = np.sum((y_vrai == cl) & (y_pred == cl))
        fp = np.sum((y_vrai != cl) & (y_pred == cl))
        fn = np.sum((y_vrai == cl) & (y_pred != cl))
        prec = vp / max(vp + fp, 1)
        rap = vp / max(vp + fn, 1)
        rappels.append(rap)
        f1.append(2 * prec * rap / max(prec + rap, 1e-12))
    return {
        "exactitude": exact,
        "exactitude_equilibree": float(np.mean(rappels)),
        "f1_macro": float(np.mean(f1)),
    }
