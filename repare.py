import os
import shutil
import json
from pathlib import Path
import torch
from datasets import Dataset

cache_dir = Path(os.path.expanduser("~/.cache/huggingface/lerobot/jogarulfo"))
src_dir = cache_dir / "dataset_MVP_store_cardboard_1"
dst_dir = cache_dir / "dataset_MVP_store_cardboard_1_fixed"

print(f"Cloning {src_dir.name}\n  to {dst_dir.name}...\n")

# 1. Copy directory
if dst_dir.exists():
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

# 2. Rename the physical video directory
src_vid = dst_dir / "videos" / "observation.images.front"
dst_vid = dst_dir / "videos" / "observation.images.wrist"
if src_vid.exists():
    src_vid.rename(dst_vid)
    print(" -> Renamed physical video folder to 'wrist'")

# 3. Fix info.json
info_file = dst_dir / "meta" / "info.json"
if info_file.exists():
    with open(info_file, 'r') as f:
        info = json.load(f)
    if 'observation.images.front' in info.get('features', {}):
        info['features']['observation.images.wrist'] = info['features'].pop('observation.images.front')
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print(" -> Fixed info.json")

# 4. Fix stats.json (if it exists)
stats_file = dst_dir / "meta" / "stats.json"
if stats_file.exists():
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    if 'observation.images.front' in stats:
        stats['observation.images.wrist'] = stats.pop('observation.images.front')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(" -> Fixed stats.json")

# 5. Fix episodes.pth (Hugging Face Dataset Object)
episodes_file = dst_dir / "meta" / "episodes.pth"
if episodes_file.exists():
    episodes_dataset = torch.load(episodes_file, weights_only=False)
    if hasattr(episodes_dataset, 'column_names'):
        cols_to_rename = {col: col.replace('observation.images.front', 'observation.images.wrist')
                          for col in episodes_dataset.column_names if 'observation.images.front' in col}
        if cols_to_rename:
            episodes_dataset = episodes_dataset.rename_columns(cols_to_rename)
            torch.save(episodes_dataset, episodes_file)
            print(" -> Fixed episodes.pth")

# 6. Fix the underlying Parquet databases
data_dir = dst_dir / "data"
if data_dir.exists():
    for parquet_file in data_dir.glob("*.parquet"):
        ds = Dataset.from_parquet(str(parquet_file))
        cols_to_rename = {col: col.replace('observation.images.front', 'observation.images.wrist')
                          for col in ds.column_names if 'observation.images.front' in col}
        if cols_to_rename:
            ds = ds.rename_columns(cols_to_rename)
            ds.to_parquet(str(parquet_file))
            print(f" -> Fixed {parquet_file.name}")

print("\nSuccess! Dataset 1 is completely repaired and saved as 'dataset_MVP_store_cardboard_1_fixed'.")