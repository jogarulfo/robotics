#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add episode-level conditioning labels to a cached LeRobot dataset.

This utility edits `meta/episodes/chunk-*/file-*.parquet` in place by adding or
updating a `conditioning` column.

Examples:

1) Use explicit mapping from a JSON file:

    python -m lerobot.scripts.lerobot_add_conditioning_labels \
        --repo-id jogarulfo/dataset_cellule_boxv2 \
        --mapping-json /path/to/conditioning_map.json

Where `conditioning_map.json` is:

    {
      "0": 8,
      "1": 7,
      "2": 6
    }

2) Reuse `task_index` as conditioning:

    python -m lerobot.scripts.lerobot_add_conditioning_labels \
        --repo-id jogarulfo/dataset_cellule_boxv2 \
        --from-task-index

3) Preview only (no file writes):

    python -m lerobot.scripts.lerobot_add_conditioning_labels \
        --repo-id jogarulfo/dataset_cellule_boxv2 \
        --mapping-json /path/to/conditioning_map.json \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lerobot.utils.constants import HF_LEROBOT_HOME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. jogarulfo/dataset_cellule_boxv2")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=HF_LEROBOT_HOME,
        help="Root of local LeRobot cache (default: HF_LEROBOT_HOME)",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="JSON file mapping episode_index (str/int) -> conditioning label (int)",
    )
    parser.add_argument(
        "--from-task-index",
        action="store_true",
        help="If set, use task_index as conditioning label for each episode.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing conditioning labels if they already exist.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary without writing files.")
    return parser.parse_args()


def load_mapping(mapping_path: Path | None) -> dict[int, int]:
    if mapping_path is None:
        return {}

    with open(mapping_path) as f:
        raw = json.load(f)

    mapping: dict[int, int] = {}
    for ep_idx, cond in raw.items():
        mapping[int(ep_idx)] = int(cond)
    return mapping


def resolve_dataset_dir(cache_root: Path, repo_id: str) -> Path:
    dataset_dir = cache_root / repo_id
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset cache path does not exist: {dataset_dir}. "
            "Make sure the dataset has been downloaded/recorded locally."
        )
    return dataset_dir


def collect_episode_files(dataset_dir: Path) -> list[Path]:
    episodes_dir = dataset_dir / "meta" / "episodes"
    files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under: {episodes_dir}")
    return files


def build_conditioning_column(
    frame: pd.DataFrame,
    mapping: dict[int, int],
    from_task_index: bool,
    overwrite_existing: bool,
) -> pd.Series:
    if "episode_index" not in frame.columns:
        raise ValueError("Missing required column 'episode_index' in episodes parquet.")

    has_existing = "conditioning" in frame.columns
    if has_existing and not overwrite_existing:
        return frame["conditioning"].astype("int64")

    if mapping:
        cond = frame["episode_index"].map(mapping)
    elif from_task_index:
        if "task_index" not in frame.columns:
            raise ValueError("--from-task-index was set but column 'task_index' was not found.")
        cond = frame["task_index"]
    else:
        raise ValueError("Provide either --mapping-json or --from-task-index.")

    missing = frame.loc[cond.isna(), "episode_index"].tolist()
    if missing:
        raise ValueError(f"Missing conditioning labels for episodes: {missing}")

    return cond.astype("int64")


def main() -> None:
    args = parse_args()

    if args.mapping_json is None and not args.from_task_index:
        raise ValueError("You must pass either --mapping-json or --from-task-index.")

    mapping = load_mapping(args.mapping_json)
    dataset_dir = resolve_dataset_dir(args.cache_root, args.repo_id)
    files = collect_episode_files(dataset_dir)

    rows_total = 0
    labels_seen: set[int] = set()

    for path in files:
        frame = pd.read_parquet(path)
        conditioning = build_conditioning_column(
            frame,
            mapping=mapping,
            from_task_index=args.from_task_index,
            overwrite_existing=args.overwrite_existing,
        )

        frame["conditioning"] = conditioning
        rows_total += len(frame)
        labels_seen.update(int(x) for x in frame["conditioning"].unique().tolist())

        if not args.dry_run:
            frame.to_parquet(path, index=False)

    mode = "DRY-RUN" if args.dry_run else "UPDATED"
    print(
        f"{mode}: {len(files)} files, {rows_total} episode rows, "
        f"conditioning labels={sorted(labels_seen)}"
    )


if __name__ == "__main__":
    main()
