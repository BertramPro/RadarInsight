"""Backfill trajectory IDs for a completed experiment's validation confusion matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_rd.train import Frame, RDCache, RDDataset, SmallRDCNN, evaluate, write_json


def read_frames(manifest_path: Path, split_name: str) -> list[Frame]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [
            Frame(row["path"], row["trajectory_id"], row["source_target"], int(row["label"]))
            for row in csv.DictReader(handle)
            if row["split"] == split_name
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    metrics_path = output_dir / "validation_best.json"
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if saved_metrics.get("confusion_cases") is not None:
        print("confusion_cases already present")
        return

    torch.set_num_threads(max(1, args.threads))
    frames = read_frames(output_dir / "manifest.csv", "val")
    cache_value = config.get("rd_cache")
    if not cache_value:
        raise ValueError("Backfill requires the experiment's completed rd_cache")
    cache = RDCache(
        Path(cache_value), frames,
        velocity_min=float(config["velocity_min"]), velocity_max=float(config["velocity_max"]),
        target_width=int(config["target_width"]), resampling=str(config["resampling"]),
    )
    dataset = RDDataset(
        frames, float(config["normalization_mean"]), float(config["normalization_std"]),
        velocity_min=float(config["velocity_min"]), velocity_max=float(config["velocity_max"]),
        target_width=int(config["target_width"]), resampling=str(config["resampling"]),
        normalization=str(config["normalization"]), input_mode=str(config["input_mode"]),
        augmentation="off", rd_cache=cache,
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    input_channels = 1 if config["input_mode"] == "rd" else 2
    model = SmallRDCNN(input_channels=input_channels, head=str(config.get("model_head", "global")))
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate(model, loader, torch.device("cpu"))
    if result["confusion_matrix"] != saved_metrics.get("confusion_matrix"):
        raise RuntimeError("Recomputed confusion matrix differs from validation_best.json; refusing to write cases")
    saved_metrics["confusion_cases"] = result["confusion_cases"]
    write_json(metrics_path, saved_metrics)
    print(json.dumps({"output_dir": str(output_dir), "error_pairs": len(result["confusion_cases"])}))


if __name__ == "__main__":
    main()
