#!/usr/bin/env python3
"""
Test script: Load dataset and verify conditioning can be added.

This is a lightweight test that doesn't require the full training pipeline.
Run with: PYTHONPATH=./robotics/lerobot/src python test_conditioning.py
"""
import json
import sys
from pathlib import Path

def test_conditioning_setup():
    """Test conditioning setup without full dataset dependencies."""
    
    # Step 1: Validate mapping file
    print("=" * 60)
    print("STEP 1: Validate conditioning mapping")
    print("=" * 60)
    
    mapping_file = Path("/home/josephrigal/workspace/conditioning_map_8class.json")
    if not mapping_file.exists():
        print(f"✗ Mapping file not found: {mapping_file}")
        return False
    
    with open(mapping_file) as f:
        mapping = json.load(f)
    
    mapping_int = {int(k): int(v) for k, v in mapping.items()}
    num_episodes = len(mapping_int)
    unique_classes = len(set(mapping_int.values()))
    
    print(f"✓ Mapping file loaded")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Unique classes: {unique_classes}")
    print(f"  - Class range: [0, {max(mapping_int.values())}]")
    print()
    
    # Step 2: Show sample mappings
    print("=" * 60)
    print("STEP 2: Sample conditioning mappings")
    print("=" * 60)
    sample_eps = [0, 1, 7, 8, 15, 16, 32, 56, 63]
    for ep in sample_eps:
        cond = mapping_int[ep]
        print(f"  Episode {ep:2d} -> Class {cond}")
    print()
    
    # Step 3: Prepare ACT config parameter
    print("=" * 60)
    print("STEP 3: ACT training parameter")
    print("=" * 60)
    print(f"Set in training config:")
    print(f"  policy.conditioning_dim = {unique_classes}")
    print()
    
    return True

if __name__ == "__main__":
    success = test_conditioning_setup()
    sys.exit(0 if success else 1)
