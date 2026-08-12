"""Persist branch-owned and fused trajectory decisions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .model import CLASS_NAMES


def classification_metrics(records: list[dict[str, object]], prediction_key: str) -> dict[str, object]:
    if not records:
        raise ValueError("cannot calculate metrics for an empty record list")
    truth = np.asarray([int(record["true_class"]) for record in records])
    prediction = np.asarray([int(record[prediction_key]) for record in records])
    cases: dict[str, list[str]] = defaultdict(list)
    for record, actual, predicted in zip(records, truth, prediction):
        if actual != predicted:
            cases[f"{actual}:{predicted}"].append(str(record["trajectory_id"]))
    return {
        "trajectory_count": len(records),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=range(5)).tolist(),
        "confusion_cases": dict(cases),
        "classification_report": classification_report(
            truth,
            prediction,
            labels=range(5),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def complementarity(records: list[dict[str, object]]) -> dict[str, object]:
    counts = defaultdict(int)
    pair_counts = defaultdict(int)
    for record in records:
        truth = int(record["true_class"])
        tr_correct = int(record["tr_prediction"]) == truth
        rd_correct = int(record["rd_prediction"]) == truth
        fused_correct = int(record["fused_prediction"]) == truth
        if tr_correct and rd_correct:
            counts["both_correct"] += 1
        elif tr_correct:
            counts["tr_only_correct"] += 1
        elif rd_correct:
            counts["rd_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
        if not tr_correct and fused_correct:
            counts["fusion_rescue_vs_tr"] += 1
        if tr_correct and not fused_correct:
            counts["fusion_harm_vs_tr"] += 1
        if int(record["tr_prediction"]) != int(record["rd_prediction"]):
            counts["branch_disagreement"] += 1
        if truth in {1, 4}:
            pair_counts["bird_other_total"] += 1
            pair_counts["bird_other_tr_correct"] += int(tr_correct)
            pair_counts["bird_other_rd_correct"] += int(rd_correct)
            pair_counts["bird_other_fused_correct"] += int(fused_correct)
    return {**counts, **pair_counts}


def save_fusion_report(
    output_dir: Path | str,
    records: list[dict[str, object]],
    provenance: dict[str, object],
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectory_decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "provenance": provenance,
        "trajectory_count": len(records),
        "tr_branch": classification_metrics(records, "tr_prediction"),
        "rd_branch": classification_metrics(records, "rd_prediction"),
        "soft_cascade": classification_metrics(records, "fused_prediction"),
        "complementarity": complementarity(records),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
