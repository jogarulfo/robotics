#!/usr/bin/env python3
"""
Verify conditioning mapping and generate summary report.
This mimics the lerobot_add_conditioning_labels logic without pandas.
"""
import json
from pathlib import Path

def validate_conditioning_mapping(mapping_path, num_expected_episodes=64):
    """Load and validate a conditioning mapping JSON."""
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    # Convert all keys to ints
    mapping_int = {}
    for ep_idx, cond in mapping.items():
        mapping_int[int(ep_idx)] = int(cond)
    
    print(f"✓ Loaded conditioning map from {mapping_path}")
    print(f"  Episodes in map: {len(mapping_int)}")
    print(f"  Expected episodes: {num_expected_episodes}")
    
    # Check coverage
    missing = set(range(num_expected_episodes)) - set(mapping_int.keys())
    if missing:
        print(f"✗ WARNING: Missing episode labels for {len(missing)} episodes: {sorted(missing)[:10]}...")
        return False
    
    # Check label range and distribution
    cond_values = sorted(set(mapping_int.values()))
    print(f"  Conditioning labels: {cond_values}")
    print(f"  Unique labels: {len(cond_values)}")
    
    # Count per label
    label_counts = {}
    for ep, cond in mapping_int.items():
        label_counts[cond] = label_counts.get(cond, 0) + 1
    
    print(f"  Distribution: {dict(sorted(label_counts.items()))}")
    
    # Show sample mappings
    print(f"\n  Sample mappings:")
    for ep in [0, 1, 7, 8, 9, 15, 16, 32, 56, 63]:
        if ep in mapping_int:
            print(f"    Episode {ep:2d} -> Class {mapping_int[ep]}")
    
    print(f"\n✓ Conditioning mapping is VALID")
    print(f"  conditioning_dim should be set to {max(cond_values) + 1}")
    return True

if __name__ == "__main__":
    mapping_file = Path("/home/josephrigal/workspace/conditioning_map_8class.json")
    if mapping_file.exists():
        validate_conditioning_mapping(mapping_file, num_expected_episodes=64)
    else:
        print(f"✗ File not found: {mapping_file}")
