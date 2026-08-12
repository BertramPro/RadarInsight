"""Traditional ML implementation for the TR branch.

The module deliberately uses the existing 22 physical track features and
provides the same trajectory-level probability contract as the DL branch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import TrajectoryDataset, TrajectoryRecord
from .model import CLASS_NAMES
from .track_features import PHYSICAL_FEATURE_COLUMNS


ML_SCHEMA = 1


def _record_identity(records: list[TrajectoryRecord]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        stat = record.csv_path.stat()
        result.append({"id": record.trajectory_id, "label": record.label,
                       "path": str(record.csv_path), "size": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns})
    return result


def feature_matrix(records: list[TrajectoryRecord], cache_root: Path | None = None,
                   *, partition: str = "train") -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Extract/cache one 22-dimensional row per trajectory."""
    identity = {"schema": ML_SCHEMA, "partition": partition,
                "columns": list(PHYSICAL_FEATURE_COLUMNS),
                "records": _record_identity(records)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_path = (cache_root or Path(__file__).resolve().parents[1] / "cache" / "ml_track") / f"{digest}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        payload = np.load(cache_path, allow_pickle=False)
        if str(payload["identity"].item()) == json.dumps(identity, sort_keys=True, ensure_ascii=False):
            return payload["x"].astype(np.float64), payload["y"].astype(np.int64), list(PHYSICAL_FEATURE_COLUMNS), {"path": str(cache_path), "key": digest, "hit": True, "records": len(records)}
    dataset = TrajectoryDataset(records)
    x = np.asarray([dataset[i].physical.numpy() for i in range(len(dataset))], dtype=np.float64)
    y = np.asarray([record.label for record in records], dtype=np.int64)
    np.savez_compressed(cache_path, x=x, y=y, identity=json.dumps(identity, sort_keys=True, ensure_ascii=False))
    return x, y, list(PHYSICAL_FEATURE_COLUMNS), {"path": str(cache_path), "key": digest, "hit": False, "records": len(records)}


def clean_train_features(x_train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...] | tuple[np.ndarray, np.ndarray]:
    finite = np.where(np.isfinite(x_train), x_train, np.nan)
    median = np.nanmedian(finite, axis=0)
    median[~np.isfinite(median)] = 0.0
    def clean(x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, median).astype(np.float64)
    cleaned = (clean(x_train), *(clean(x) for x in others))
    return cleaned + (median,)


def inverse_frequency_weights(y: np.ndarray, classes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    weights = len(y) / (classes * np.maximum(counts, 1.0))
    sample = weights[y]
    sample /= max(float(sample.mean()), 1e-12)
    return sample, weights


def build_model(kind: str, seed: int):
    kind = str(kind)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass", num_class=5, n_estimators=400,
                              learning_rate=0.04, num_leaves=15, min_child_samples=12,
                              reg_alpha=0.1, reg_lambda=2.0, random_state=seed,
                              n_jobs=-1, verbosity=-1)
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300,
                                              max_leaf_nodes=31, min_samples_leaf=8,
                                              l2_regularization=0.1, random_state=seed)
    if kind == "extra_trees":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                    max_features="sqrt", class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
    if kind == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale",
                            class_weight="balanced", probability=True,
                            random_state=seed))
    raise ValueError(f"unsupported TR ML model: {kind}")


class WeightedSoftVoting:
    """Small serializable soft-voting wrapper with a stable class contract."""
    def __init__(self, members: list[str], weights: list[float], seed: int):
        if len(members) < 2 or len(members) != len(weights) or any(float(w) <= 0 for w in weights):
            raise ValueError("soft voting needs at least two models and positive weights")
        self.members, self.weights, self.seed = list(members), [float(w) for w in weights], int(seed)
        self.models = [build_model(name, seed) for name in self.members]

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        for model in self.models:
            if sample_weight is not None and model.__class__.__name__ == "LGBMClassifier":
                model.fit(x, y, sample_weight=sample_weight)
            elif sample_weight is not None and isinstance(model, HistGradientBoostingClassifier):
                model.fit(x, y, sample_weight=sample_weight)
            else:
                model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = np.stack([model.predict_proba(x) for model in self.models], axis=0)
        return np.average(probs, axis=0, weights=np.asarray(self.weights, dtype=np.float64))


def save_bundle(path: Path, *, model: Any, model_kind: str, feature_names: list[str],
                median: np.ndarray, config: dict[str, Any]) -> None:
    bundle = {"schema": ML_SCHEMA, "implementation": "ml", "model": model,
              "model_kind": model_kind, "feature_names": feature_names,
              "median": median.tolist(), "class_names": list(CLASS_NAMES),
              "config": config}
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("implementation") != "ml":
        raise ValueError(f"not a TR ML checkpoint: {path}")
    return bundle

