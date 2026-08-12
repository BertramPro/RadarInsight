"""Train the traditional-ML implementation of the TR branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_fusion.data import load_grouped_split, load_trajectory_records  # noqa: E402
from radar_fusion.ml_track import (  # noqa: E402
    WeightedSoftVoting, build_model, clean_train_features, feature_matrix,
    inverse_frequency_weights, save_bundle,
)
from radar_fusion.model import CLASS_NAMES  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--track-index", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=["lightgbm", "hist_gradient_boosting"])
    p.add_argument("--model-weights", nargs="+", type=float, default=None)
    p.add_argument("--model-plan", default="custom")
    p.add_argument("--soft-voting", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def metrics(y, prob):
    pred = np.asarray(prob).argmax(axis=1)
    report = classification_report(y, pred, labels=np.arange(5), target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {"trajectory_count": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(5)).tolist(),
            "classification_report": report}


def write_progress(output, phase, percent, **extra):
    payload = {"phase": phase, "percent": float(percent), **extra}
    (output / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if len(args.models) == 0: raise ValueError("at least one TR ML model is required")
    if args.model_weights is not None and len(args.model_weights) != len(args.models):
        raise ValueError("model weights must match model count")
    weights = args.model_weights or [1.0] * len(args.models)
    split = load_grouped_split(args.split)
    records = {part: load_trajectory_records(args.track_index, split, part) for part in ("train", "val", "test")}
    write_progress(output, "preparing", 5.0, stage="extracting_22d_features")
    matrices = {}
    caches = {}
    for index, part in enumerate(("train", "val", "test"), start=1):
        x, y, names, cache = feature_matrix(records[part], partition=part)
        matrices[part] = [x, y]; caches[part] = cache
        write_progress(output, "preparing", 5.0 + index * 10.0, stage=f"features_{part}")
    x_train, x_val, x_test, median = clean_train_features(matrices["train"][0], matrices["val"][0], matrices["test"][0])
    y_train, y_val, y_test = matrices["train"][1], matrices["val"][1], matrices["test"][1]
    sample_weight, class_weights = inverse_frequency_weights(y_train)
    config = {"experiment_type": "tr_only", "tr_implementation": "ml", "seed": args.seed,
              "ml_models": args.models, "ml_model_weights": weights,
              "ml_soft_voting": bool(args.soft_voting), "ml_model_plan": args.model_plan,
              "models": args.models, "model_weights": weights, "feature_count": len(names),
              "feature_names": names, "class_names": list(CLASS_NAMES), "split": {
                  "manifest": str(args.split.resolve()), "sha256": sha256(args.split),
                  "train": len(records["train"]), "val": len(records["val"]), "test": len(records["test"])},
              "test_deferred": bool(args.skip_test), "class_weight_rule": "normalized_inverse_frequency",
              "class_weights": class_weights.tolist(), "feature_cache": caches}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidates = {}
    fitted = {}
    for index, kind in enumerate(args.models):
        write_progress(output, "train", 20.0 + index * 55.0 / len(args.models), model=kind, candidate=index + 1, total_candidates=len(args.models))
        model = build_model(kind, args.seed)
        if kind in {"lightgbm", "hist_gradient_boosting"}:
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_train)
        prob = model.predict_proba(x_val); candidates[kind] = {"validation": metrics(y_val, prob)}; fitted[kind] = model
    if args.soft_voting:
        if len(args.models) < 2:
            raise ValueError("soft voting needs at least two member models")
        members = list(args.models)
        model = WeightedSoftVoting(members, weights, args.seed).fit(x_train, y_train, sample_weight)
        prob = model.predict_proba(x_val); candidates["soft_voting"] = {"validation": metrics(y_val, prob)}; fitted["soft_voting"] = model
    best_name = max(candidates, key=lambda name: candidates[name]["validation"]["macro_f1"])
    best_model = fitted[best_name]
    best_validation = candidates[best_name]["validation"]
    decisions = []
    for record, prob in zip(records["val"], best_model.predict_proba(x_val)):
        pred = int(np.argmax(prob)); decisions.append({"trajectory_id": record.trajectory_id, "true_class": record.label,
            "true_label": CLASS_NAMES[record.label], "tr_probabilities": prob.tolist(), "prediction": pred,
            "prediction_label": CLASS_NAMES[pred], "implementation": "ml"})
    (output / "trajectory_decisions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in decisions), encoding="utf-8")
    (output / "validation_best.json").write_text(json.dumps(best_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "candidate_metrics.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        importance = permutation_importance(best_model, x_val, y_val, scoring="f1_macro", n_repeats=5, random_state=args.seed, n_jobs=-1)
        order = np.argsort(importance.importances_mean)[::-1]
        (output / "feature_importance.json").write_text(json.dumps([{"feature": names[i], "mean_decrease_macro_f1": float(importance.importances_mean[i]), "std": float(importance.importances_std[i])} for i in order], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        (output / "feature_importance.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_bundle(output / "best_model.joblib", model=best_model, model_kind=best_name, feature_names=names, median=median, config=config)
    config["selected_model"] = best_name
    (output / "model_metadata.json").write_text(json.dumps({"implementation": "ml", "model_kind": best_name, "feature_names": names, "class_names": list(CLASS_NAMES)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.skip_test:
        test_metrics = metrics(y_test, best_model.predict_proba(x_test)); (output / "test_trajectory_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(output, "complete", 100.0, selected_model=best_name)


if __name__ == "__main__": main()
