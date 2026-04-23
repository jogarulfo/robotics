import torch
import json
import os
from pathlib import Path

cache_dir = Path(os.path.expanduser("~/.cache/huggingface/lerobot/jogarulfo"))
repo_name = "dataset_MVP_store_cardboard_1"
meta_dir = cache_dir / repo_name / "meta"

print(f"--- Diagnosing {repo_name} ---")

# 1. Inspect info.json
info_file = meta_dir / "info.json"
print("\n[1] Inspecting info.json...")
if info_file.exists():
    with open(info_file, 'r') as f:
        info = json.load(f)
        if 'features' in info:
            print("Camera/State features mapped in info.json:")
            for k in info['features'].keys():
                if 'observation' in k:
                    print(f"  - {k}")
else:
    print("  -> info.json not found!")

# 2. Inspect episodes.pth
episodes_file = meta_dir / "episodes.pth"
print("\n[2] Inspecting episodes.pth...")
if episodes_file.exists():
    episodes = torch.load(episodes_file, weights_only=False)
    print(f"  -> Base Data Type: {type(episodes)}")
    
    # If it's a Dictionary
    if isinstance(episodes, dict):
        print(f"  -> Number of items: {len(episodes)}")
        first_key = list(episodes.keys())[0]
        first_val = episodes[first_key]
        print(f"  -> Type of items inside dict: {type(first_val)}")
        
        if hasattr(first_val, 'keys'):
            print("\n  -> Keys inside the first item (filtering for 'observation' or 'videos'):")
            for k in sorted(first_val.keys()):
                if 'observation' in k or 'videos' in k:
                    val_type = type(first_val[k]).__name__
                    print(f"      - '{k}' (Type: {val_type})")
                    
    # If it's a List
    elif isinstance(episodes, list):
        print(f"  -> List Length: {len(episodes)}")
        if len(episodes) > 0:
            first_val = episodes[0]
            print(f"  -> Type of items inside list: {type(first_val)}")
            
            if hasattr(first_val, 'keys'):
                print("\n  -> Keys inside the first item (filtering for 'observation' or 'videos'):")
                for k in sorted(first_val.keys()):
                    if 'observation' in k or 'videos' in k:
                        val_type = type(first_val[k]).__name__
                        print(f"      - '{k}' (Type: {val_type})")
    else:
        # If it's some custom LeRobot class or Hugging Face dataset
        print("\n  -> UNKNOWN STRUCTURE. Checking attributes...")
        print(f"  -> Attributes: {dir(episodes)}")

else:
    print("  -> episodes.pth not found!")