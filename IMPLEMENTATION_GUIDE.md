# ACT Conditioning Implementation for Dataset MVP Store Cardboard

## Executive Summary

You now have everything set up to train ACT with spatial conditioning on your 8×8 cardboard picking dataset. The conditioning layer allows the model to learn position-dependent manipulation strategies (e.g., "when at position 0, pick the top-left box; when at position 1, pick one step right").

**Key insight**: Episodes 0, 8, 16, 32... → class 0 (same initial behavior)
**Architecture**: 8 spatial conditioning classes mapping to `conditioning_dim=8` in ACT

---

## What Has Been Prepared

### 1. **conditioning_map_8class.json**
- Maps 64 episodes to 8 conditioning classes
- Structure: `episode_index mod 8` = conditioning class
- Balanced: 8 episodes per class
- Location: `/home/josephrigal/workspace/conditioning_map_8class.json`

**Sample mappings:**
```
Episode 0 → Class 0    (pick position 0, first 8-box sequence)
Episode 1 → Class 1    (pick position 1, first 8-box sequence)
Episode 8 → Class 0    (pick position 0, SECOND 8-box sequence - same behavior!)
Episode 16 → Class 0   (pick position 0, THIRD 8-box sequence - same behavior!)
```

### 2. **Validation Tools**
- `test_conditioning.py` - Validates mapping structure (no dependencies)
- `validate_conditioning_map.py` - Detailed distribution analysis
- `conditioning_study_guide.py` - Code reference for understanding the layers

### 3. **Setup & Documentation**
- `setup_conditioning.sh` - Complete step-by-step guide
- `README_conditioning.md` - Technical deep-dive
- `conditioning_study_guide.py` - ACT code flow explanation
- `add_conditioning_to_dataset.py` - Apply mapping to dataset (requires pandas)

---

## Execution Plan: 3 Steps

### **STEP 1: Add Conditioning Labels to Dataset**

```bash
cd /home/josephrigal/workspace/robotics

# Install required dependency (if not already installed)
pip install pandas

# Dry-run first (no changes, just validation)
PYTHONPATH=./src python3 -m lerobot.scripts.lerobot_add_conditioning_labels \
  --repo-id jogarulfo/dataset_MVP_store_cardboard \
  --mapping-json /home/josephrigal/workspace/conditioning_map_8class.json \
  --dry-run

# Output should show:
# "DRY-RUN: 8 files, 64 episode rows, conditioning labels=[0, 1, 2, 3, 4, 5, 6, 7]"
```

Once verified, write to dataset:
```bash
PYTHONPATH=./src python3 -m lerobot.scripts.lerobot_add_conditioning_labels \
  --repo-id jogarulfo/dataset_MVP_store_cardboard \
  --mapping-json /home/josephrigal/workspace/conditioning_map_8class.json \
  --overwrite-existing
```

**What this does:**
- Reads `~/.cache/huggingface/lerobot/jogarulfo/dataset_MVP_store_cardboard/meta/episodes/` parquet files
- Injects `conditioning` column with values 0-7
- Writes modified metadata back to disk

### **STEP 2: Train ACT with Conditioning**

```bash
cd /home/josephrigal/workspace/robotics

# Minimal working example (5K steps)
python3 -m lerobot.scripts.lerobot_train \
  --dataset-repo-id jogarulfo/dataset_MVP_store_cardboard \
  --policy.type act \
  --policy.conditioning_dim 8 \
  --batch-size 32 \
  --steps 5000 \
  --log-freq 100 \
  --eval-freq 500 \
  --output-dir ./outputs/act_with_conditioning_8class
```

**Key parameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| `policy.type` | `act` | Use Action Chunking Transformer |
| `policy.conditioning_dim` | `8` | Enable conditioning with 8 classes |
| `batch-size` | `32` | Standard batch size |
| `steps` | `5000` | Short initial run to validate |
| `log-freq` | `100` | Print stats every 100 updates |
| `eval-freq` | `500` | Evaluate every 500 updates |

### **STEP 3: Validate Training & Compare with Baseline**

```bash
# Train without conditioning (for comparison)
python3 -m lerobot.scripts.lerobot_train \
  --dataset-repo-id jogarulfo/dataset_MVP_store_cardboard \
  --policy.type act \
  --batch-size 32 \
  --steps 5000 \
  --output-dir ./outputs/act_no_conditioning
```

**Comparison metrics to track:**
- Training loss convergence speed
- Validation action MSE
- Generalization to held-out episodes
- Final checkpoint performance

---

## How the Conditioning Layer Works (Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│          Dataset: Load Frame (frame_idx=42)               │
│  episode_idx = 16 (from frame metadata)                    │
│  Lookup: _episode_conditioning[16] = 0                     │
│  Return: {..., "conditioning": 0}                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│     DataLoader: Collate Batch (32 frames)                  │
│  batch["conditioning"] = [0, 1, 2, 3, 4, 5, 6, 7,        │
│                           0, 1, 2, 3, ...]  shape: (32,)   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              ACT.forward(batch)                             │
│  1. conditioning = batch["conditioning"]  # (32,)           │
│  2. cond_embed = self.conditioning_proj(conditioning)       │
│     # Maps 0-7 → (32, 512) dense vectors                    │
│  3. encoder_in_tokens.append(cond_embed)                    │
│     # Now encoder sees: [latent, robot_state,               │
│     #                    env_state, CONDITIONING, images]  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│      Transformer Encoder (learns position-specific          │
│      action strategies based on conditioning)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│      Decoder + Action Head                                 │
│      Output: action_predictions  # (32, 100, 4)            │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure & Locations

```
/home/josephrigal/workspace/
├── conditioning_map_8class.json           # Episode → class (JSON)
├── test_conditioning.py                   # Quick validation (no deps)
├── validate_conditioning_map.py           # Detailed analysis
├── add_conditioning_to_dataset.py         # Apply to dataset
├── setup_conditioning.sh                  # Step-by-step guide (bash)
├── README_conditioning.md                 # Technical documentation
├── conditioning_study_guide.py            # Code reference & examples
└── IMPLEMENTATION_GUIDE.md                # This file

/home/josephrigal/workspace/robotics/
├── lerobot/src/lerobot/
│   ├── policies/act/
│   │   ├── modeling_act.py                # Line 354-482: conditioning
│   │   └── configuration_act.py           # Line 89: conditioning_dim
│   ├── datasets/
│   │   └── lerobot_dataset.py             # Line 1084-1089: loads conditioning
│   └── scripts/
│       ├── lerobot_train.py               # Training entry point
│       └── lerobot_add_conditioning_labels.py  # Applies mapping
```

---

## Debugging Checklist

### ✓ Pre-Training Checks

- [ ] Mapping file exists: `ls /home/josephrigal/workspace/conditioning_map_8class.json`
- [ ] Mapping is valid: `python3 test_conditioning.py` (shows 8 classes, 8 episodes each)
- [ ] Dataset is cached: `ls ~/.cache/huggingface/lerobot/jogarulfo/`
- [ ] Pandas installed: `python3 -c "import pandas; print(pandas.__version__)"`

### ✓ After Running Step 1 (adding conditioning)

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("jogarulfo/dataset_MVP_store_cardboard")
for i in range(5):
    sample = ds[i]
    print(f"Frame {i}: conditioning = {sample.get('conditioning', 'MISSING')}")
    # Should output: conditioning = 0, 1, 2, 3, 4, ... (not "MISSING")
```

### ✓ During Training

```python
# Add to lerobot_train.py for debugging (around line 390)
if step == 0:
    batch = next(dl_iter)
    print(f"DEBUG: batch keys = {batch.keys()}")
    if "conditioning" in batch:
        print(f"✓ conditioning shape: {batch['conditioning'].shape}")
        print(f"  values: {batch['conditioning'][:8]}")
    else:
        print("✗ WARNING: conditioning not in batch!")
```

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'datasets'` | `pip install datasets huggingface-hub` |
| `ModuleNotFoundError: No module named 'pandas'` | `pip install pandas` |
| `ValueError: conditioning must be present in batch` | Step 1 (adding conditioning) wasn't run, or failed silently |
| `FileNotFoundError: Dataset cache path does not exist` | Dataset not cached; load once: `LeRobotDataset("jogarulfo/dataset_MVP_store_cardboard")` |
| Loss is NaN after few steps | Likely unrelated to conditioning; check learning rate, data normalization |

---

## Expected Results

### Baseline (No Conditioning)
- Single policy learns average picking behavior for all positions
- May struggle with position-specific strategies
- **Loss:** ~0.05-0.10 after 5K steps

### With 8-Class Conditioning
- Policy learns 8 distinct sub-policies (one per position)
- Better generalization to new box arrangements
- **Loss:** Should be similar or lower than baseline
- **Benefit visible in:** Position-sensitive action patterns

---

## Next: Extended Training

Once validated, scale up:

```bash
python3 -m lerobot.scripts.lerobot_train \
  --dataset-repo-id jogarulfo/dataset_MVP_store_cardboard \
  --policy.type act \
  --policy.conditioning_dim 8 \
  --batch-size 32 \
  --steps 100000 \
  --log-freq 1000 \
  --eval-freq 10000 \
  --save-freq 10000 \
  --output-dir ./outputs/act_conditioning_large
```

---

## References

- **ACT Paper**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://huggingface.co/papers/2304.13705)
- **LeRobot Docs**: [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- **Code Files**:
  - [modeling_act.py](robotics/lerobot/src/lerobot/policies/act/modeling_act.py)
  - [configuration_act.py](robotics/lerobot/src/lerobot/policies/act/configuration_act.py)
  - [lerobot_dataset.py](robotics/lerobot/src/lerobot/datasets/lerobot_dataset.py)
  - [lerobot_add_conditioning_labels.py](robotics/lerobot/src/lerobot/scripts/lerobot_add_conditioning_labels.py)

---

## Summary of Generated Files

| File | Purpose | Status |
|------|---------|--------|
| `conditioning_map_8class.json` | Episode-to-class mapping | ✓ Ready to use |
| `test_conditioning.py` | Validation (no deps) | ✓ Ready to run |
| `README_conditioning.md` | Technical docs | ✓ Ready to read |
| `setup_conditioning.sh` | Step-by-step guide | ✓ Ready to follow |
| `conditioning_study_guide.py` | Code reference | ✓ Ready to study |
| `add_conditioning_to_dataset.py` | Apply to dataset | ✓ Ready (needs pandas) |

---

**Status**: All planning and preparation complete. Ready for: **STEP 1 - Add conditioning labels**.
