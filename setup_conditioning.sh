#!/usr/bin/env bash
# Step-by-step guide to add conditioning to dataset and train ACT

set -e

REPO_ID="jogarulfo/dataset_MVP_store_cardboard"
MAPPING_FILE="/home/josephrigal/workspace/conditioning_map_8class.json"
WORKSPACE="/home/josephrigal/workspace"
LEROBOT_SRC="${WORKSPACE}/robotics/lerobot/src"

echo "============================================================"
echo "ACT Conditioning Setup Guide"
echo "============================================================"
echo ""
echo "This script adds conditioning labels to your dataset and"
echo "prepares an ACT training run with 8-class spatial conditioning."
echo ""

# Step 1: Verify mapping
echo "STEP 1: Verify conditioning mapping"
echo "----------------------------------------"
if [ ! -f "$MAPPING_FILE" ]; then
    echo "✗ Mapping file not found: $MAPPING_FILE"
    exit 1
fi
echo "✓ Mapping file exists"
echo "  Use: python test_conditioning.py"
python3 "$WORKSPACE/test_conditioning.py"

# Step 2: Add conditioning to dataset (WITH PANDAS)
echo ""
echo "STEP 2: Add conditioning labels to dataset"
echo "----------------------------------------"
echo "Command (requires pandas installed):"
echo ""
echo "PYTHONPATH=$LEROBOT_SRC python3 -m lerobot.scripts.lerobot_add_conditioning_labels \\"
echo "  --repo-id $REPO_ID \\"
echo "  --mapping-json $MAPPING_FILE \\"
echo "  --dry-run"
echo ""
echo "Then run with --overwrite-existing to write:"
echo ""
echo "PYTHONPATH=$LEROBOT_SRC python3 -m lerobot.scripts.lerobot_add_conditioning_labels \\"
echo "  --repo-id $REPO_ID \\"
echo "  --mapping-json $MAPPING_FILE \\"
echo "  --overwrite-existing"
echo ""

# Step 3: Show training command
echo "STEP 3: Train ACT with conditioning"
echo "----------------------------------------"
echo "Example training command:"
echo ""
echo "cd $WORKSPACE/robotics"
echo "PYTHONPATH=$LEROBOT_SRC python3 -m lerobot.scripts.lerobot_train \\"
echo "  --dataset-repo-id $REPO_ID \\"
echo "  --policy.type act \\"
echo "  --policy.conditioning_dim 8 \\"
echo "  --batch-size 32 \\"
echo "  --steps 1000 \\"
echo "  --log-freq 100 \\"
echo "  --eval-freq 500"
echo ""

echo "============================================================"
echo "Key Configuration Parameters:"
echo "============================================================"
echo "  conditioning_dim = 8   (number of position classes: 0-7)"
echo "  Each episode gets a label from 0-7 based on position"
echo "  Episodes 0,8,16,24,... get label 0 (first position)"
echo "  Episodes 1,9,17,25,... get label 1 (second position)"
echo "  ... and so on"
echo ""

echo "============================================================"
echo "Files Generated:"
echo "============================================================"
echo "✓ $MAPPING_FILE - Conditioning class mapping (JSON)"
echo "✓ test_conditioning.py - Validation script"
echo "✓ add_conditioning_to_dataset.py - Apply to dataset (requires pandas)"
echo ""
