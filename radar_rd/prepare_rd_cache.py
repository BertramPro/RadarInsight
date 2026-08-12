"""Precompute velocity-aligned RD frames for fast, repeatable training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Keep this script alongside train.py so it uses exactly the same RD loader and
# trajectory split implementation as training.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    COMMON_VR_MAX,
    COMMON_VR_MIN,
    DEFAULT_TRAIN_REGISTRY,
    TARGET_VR_WIDTH,
    build_manifest,
    load_rd,
    registry_train_split,
    sample_training_frames,
    split_frames,
    stratified_trajectory_split,
)


CACHE_VERSION = 1


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # The local monitor polls ``state.json`` while this long-running builder
    # updates it.  Windows may briefly deny an atomic replacement when a
    # reader has the file open, so retry rather than discarding hours of
    # already-materialized memmap data.
    last_error = None
    for _ in range(30):
        try:
            os.replace(str(temporary), str(path))
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--velocity-min", type=float, default=COMMON_VR_MIN)
    parser.add_argument("--velocity-max", type=float, default=COMMON_VR_MAX)
    parser.add_argument("--target-width", type=int, default=TARGET_VR_WIDTH)
    parser.add_argument("--resampling", choices=["db_linear", "power_linear", "area"], default="db_linear")
    parser.add_argument("--split-mode", choices=["fixed_grouped", "random_stratified", "registry_train"], default="registry_train")
    parser.add_argument("--train-registry", type=Path, default=DEFAULT_TRAIN_REGISTRY)
    parser.add_argument("--grouped-split", type=Path, default=None,
                        help="Authoritative trajectory split manifest; required for fixed_grouped")
    parser.add_argument("--max-train-frames-per-trajectory", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-test", action="store_true", help="Also cache the held-out test partition")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.velocity_min >= args.velocity_max or args.target_width < 8:
        raise ValueError("Invalid velocity interval or target width")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    frames = build_manifest(args.dataset_root.expanduser().resolve())
    if args.split_mode == "fixed_grouped":
        if args.grouped_split is None:
            raise ValueError("--grouped-split is required for fixed_grouped")
        grouped_split_path = args.grouped_split.expanduser().resolve()
        raw_split = json.loads(grouped_split_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw_split, dict) or not raw_split:
            raise ValueError(f"Grouped split is empty or invalid: {grouped_split_path}")
        group_keys = {"train": "train_group_ids", "val": "val_group_ids", "test": "test_group_ids"}
        if all(key in raw_split for key in group_keys.values()):
            split = {
                str(trajectory_id): partition
                for partition, key in group_keys.items()
                for trajectory_id in raw_split[key]
            }
        else:
            split = {str(key): str(value) for key, value in raw_split.items()}
        unsupported = set(split.values()) - {"train", "val", "test"}
        if unsupported:
            raise ValueError(f"Grouped split contains unsupported partitions: {sorted(unsupported)}")
        dataset_ids = {frame.trajectory_id for frame in frames}
        missing_ids = sorted(set(split) - dataset_ids)
        if missing_ids:
            raise ValueError(f"Grouped split contains trajectories absent from RD data (e.g. {missing_ids[0]})")
        frames = [frame for frame in frames if frame.trajectory_id in split]
        split_metadata = {
            "mode": "fixed_grouped",
            "manifest": str(grouped_split_path),
            "sha256": hashlib.sha256(grouped_split_path.read_bytes()).hexdigest(),
            "trajectory_count": len(split),
        }
    elif args.split_mode == "registry_train":
        if args.train_registry is None:
            raise ValueError("--train-registry is required for registry_train")
        split, split_metadata = registry_train_split(frames, args.train_registry.expanduser().resolve(), args.seed)
    else:
        split = stratified_trajectory_split(frames, args.seed)
        split_metadata = {"mode": "random_stratified", "train_validation_test_ratio": [0.70, 0.15, 0.15], "seed": args.seed}
    partitions = split_frames(frames, split)
    selected_train = sample_training_frames(partitions["train"], args.max_train_frames_per_trajectory, args.seed)
    used_frames = [*selected_train, *partitions["val"]]
    if args.include_test:
        used_frames.extend(partitions["test"])
    cache_frames = sorted({frame.path: frame for frame in used_frames}.values(), key=lambda x: x.path)
    index = {frame.path: position for position, frame in enumerate(cache_frames)}
    preprocessing = {
        "velocity_min": float(args.velocity_min), "velocity_max": float(args.velocity_max),
        "target_width": int(args.target_width), "resampling": str(args.resampling),
    }
    metadata_path = args.cache_dir / "metadata.json"
    index_path = args.cache_dir / "index.json"
    complete_path = args.cache_dir / "complete.json"
    images_path = args.cache_dir / "images.npy"
    observed_path = args.cache_dir / "observed.npy"
    state_path = args.cache_dir / "state.json"
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    expected_identity = {
        "split": split_metadata,
        "max_train_frames_per_trajectory": int(args.max_train_frames_per_trajectory),
        "include_test": bool(args.include_test),
    }
    if existing and (existing.get("preprocessing") != preprocessing or
                     existing.get("cache_identity") != expected_identity):
        raise ValueError("Cache directory contains a different preprocessing or split configuration")
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("cache_version") == CACHE_VERSION and complete.get("frame_count") == len(cache_frames):
            print(json.dumps({"status": "already_complete", "cache_dir": str(args.cache_dir)}, ensure_ascii=False), flush=True)
            return
    write_json(metadata_path, {"cache_version": CACHE_VERSION, "preprocessing": preprocessing,
                               "dataset_root": str(args.dataset_root), "split": split_metadata,
                               "cache_identity": expected_identity,
                               "frame_count": len(cache_frames), "train_frame_count": len(selected_train),
                               "validation_frame_count": len(partitions["val"]), "test_included": args.include_test,
                               "source_files": [{"path": frame.path, "size": Path(frame.path).stat().st_size,
                                                 "mtime_ns": Path(frame.path).stat().st_mtime_ns} for frame in cache_frames]})
    write_json(index_path, index)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    start = int(state.get("processed", 0)) if state.get("cache_version") == CACHE_VERSION else 0
    shape = tuple(state.get("shape", []))
    if not images_path.is_file() or not observed_path.is_file():
        start = 0
    if start < 0 or start > len(cache_frames):
        start = 0
    if not shape:
        sample, _ = load_rd(cache_frames[0].path, args.velocity_min, args.velocity_max, args.target_width, args.resampling)
        shape = (len(cache_frames), *sample.shape)
    images = np.lib.format.open_memmap(images_path, mode="r+" if images_path.exists() else "w+",
                                       dtype=np.float32, shape=shape)
    observed = np.lib.format.open_memmap(observed_path, mode="r+" if observed_path.exists() else "w+",
                                         dtype=np.float32, shape=(len(cache_frames), args.target_width))
    started = time.time()
    for position in range(start, len(cache_frames)):
        array, mask = load_rd(cache_frames[position].path, args.velocity_min, args.velocity_max,
                              args.target_width, args.resampling)
        if array.shape != shape[1:]:
            raise ValueError(f"RD shape changed at {cache_frames[position].path}: {array.shape} != {shape[1:]}")
        images[position] = array
        observed[position] = mask
        if (position + 1) % 100 == 0 or position + 1 == len(cache_frames):
            images.flush(); observed.flush()
            write_json(state_path, {"cache_version": CACHE_VERSION, "processed": position + 1,
                                    "total": len(cache_frames), "shape": list(shape),
                                    "updated_at": time.time()})
            elapsed = max(time.time() - started, 1e-6)
            rate = (position + 1 - start) / elapsed
            print(json.dumps({"status": "building", "processed": position + 1,
                              "total": len(cache_frames), "percent": 100.0 * (position + 1) / len(cache_frames),
                              "rate_fps": rate, "eta_seconds": (len(cache_frames) - position - 1) / max(rate, 1e-6)}, ensure_ascii=False), flush=True)
    images.flush(); observed.flush()
    write_json(complete_path, {"cache_version": CACHE_VERSION, "status": "complete",
                               "frame_count": len(cache_frames), "shape": list(shape),
                               "completed_at": time.time()})
    print(json.dumps({"status": "complete", "cache_dir": str(args.cache_dir),
                      "frame_count": len(cache_frames), "shape": list(shape)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
