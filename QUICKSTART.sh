#!/usr/bin/env bash
# QUICK START: ACT Conditioning Setup
# Follow these commands in order

set -e

WORKSPACE="/home/josephrigal/workspace"
ROBOTICS_ROOT="$WORKSPACE/robotics"
LEROBOT_SRC="$ROBOTICS_ROOT/lerobot/src"
REPO_ID="jogarulfo/dataset_MVP_store_cardboard"
MAPPING_FILE="$WORKSPACE/conditioning_map_8class.json"

echo "=========================================="
echo "ACT CONDITIONING - QUICK START"
echo "=========================================="
echo ""

# ==========================================
# CHECK 1: Validation
# ==========================================
echo "📋 CHECK 1: Validate conditioning mapping"
echo "────────────────────────────────────────"
python3 "$WORKSPACE/test_conditioning.py"
echo ""

# ==========================================
# STEP 1: Add conditioning to dataset
# ==========================================
echo "📝 STEP 1: Add conditioning labels to dataset"
echo "────────────────────────────────────────"
echo "First, ensure pandas is installed:"
echo "  pip install pandas"
echo ""
echo "Then run DRY-RUN (no changes):"
echo ""
echo "  cd $ROBOTICS_ROOT"
echo "  PYTHONPATH=$LEROBOT_SRC python3 -m lerobot.scripts.lerobot_add_conditioning_labels \\"
echo "    --repo-id $REPO_ID \\"
echo "    --mapping-json $MAPPING_FILE \\"
echo "    --dry-run"
echo ""
echo "If dry-run looks good, apply with:"
echo ""
echo "  PYTHONPATH=$LEROBOT_SRC python3 -m lerobot.scripts.lerobot_add_conditioning_labels \\"
echo "    --repo-id $REPO_ID \\"
echo "    --mapping-json $MAPPING_FILE \\"
echo "    --overwrite-existing"
echo ""

# ==========================================
# STEP 2: Train with conditioning
# ==========================================
echo "⚙️  STEP 2: Train ACT with conditioning"
echo "────────────────────────────────────────"
echo "Quick validation run (5K steps):"
echo ""
echo "  cd $ROBOTICS_ROOT"
echo "  python3 -m lerobot.scripts.lerobot_train \\"
echo "    --dataset-repo-id $REPO_ID \\"
echo "    --policy.type act \\"
echo "    --policy.conditioning_dim 8 \\"
echo "    --batch-size 32 \\"
echo "    --steps 5000 \\"
echo "    --log-freq 100 \\"
echo "    --eval-freq 500 \\"
echo "    --output-dir ./outputs/act_conditioning_test"
echo ""

# ==========================================
# STEP 3: Comparison run (optional)
# ==========================================
echo "🔍 STEP 3 (OPTIONAL): Train without conditioning (for comparison)"
echo "────────────────────────────────────────"
echo ""
echo "  python3 -m lerobot.scripts.lerobot_train \\"
echo "    --dataset-repo-id $REPO_ID \\"
echo "    --policy.type act \\"
echo "    --batch-size 32 \\"
echo "    --steps 5000 \\"
echo "    --log-freq 100 \\"
echo "    --eval-freq 500 \\"
echo "    --output-dir ./outputs/act_no_conditioning"
echo ""
echo "Then compare loss curves in outputs/"
echo ""

# ==========================================
# Full training
# ==========================================
echo "🚀 FULL TRAINING (after validation):"
echo "────────────────────────────────────────"
echo ""
echo "  python3 -m lerobot.scripts.lerobot_train \\"
echo "    --dataset-repo-id $REPO_ID \\"
echo "    --policy.type act \\"
echo "    --policy.conditioning_dim 8 \\"
echo "    --batch-size 32 \\"
echo "    --steps 100000 \\"
echo "    --log-freq 1000 \\"
echo "    --eval-freq 10000 \\"
echo "    --save-freq 10000 \\"
echo "    --output-dir ./outputs/act_conditioning_final"
echo ""

# ==========================================
# Debugging
# ==========================================
echo "🐛 DEBUGGING: Verify conditioning is in batch"
echo "────────────────────────────────────────"
echo ""
cat > /tmp/test_batch_conditioning.py << 'PYTHON'
import sys
sys.path.insert(0, '$LEROBOT_SRC')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

# Load dataset
ds = LeRobotDataset("$REPO_ID")
print(f"✓ Dataset loaded: {ds.num_episodes} episodes, {ds.num_frames} frames")

# Check individual samples
sample = ds[0]
if "conditioning" in sample:
    print(f"✓ conditioning in sample: {sample['conditioning']}")
else:
    print("✗ conditioning NOT in sample")

# Check batch
dl = DataLoader(ds, batch_size=4)
batch = next(iter(dl))
if "conditioning" in batch:
    print(f"✓ conditioning in batch: shape {batch['conditioning'].shape}")
    print(f"  values: {batch['conditioning']}")
else:
    print("✗ conditioning NOT in batch")
PYTHON

echo "  cd /tmp && python3 test_batch_conditioning.py"
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Files you'll use:"
echo "  • conditioning_map_8class.json"
echo "  • README_conditioning.md"
echo "  • IMPLEMENTATION_GUIDE.md"
echo ""
echo "Key parameters:"
echo "  • policy.conditioning_dim = 8"
echo "  • Mapping: episode % 8 = class"
echo ""
echo "Next: Follow STEP 1-2 above"
echo ""
