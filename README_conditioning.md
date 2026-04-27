# ACT Conditioning Layer Guide

## Overview

Your ACT model setup includes a **task/spatial conditioning system** that allows the model to learn different behaviors based on robot position or task state. For your 8×8 cardboard manipulation dataset, this means the model can learn to adapt its picking strategy based on which box position (0-7) it's picking from.

## How Motion Conditioning Works in ACT

### 1. **Conditioning Embedding Layer** (in the transformer encoder)

```python
# From modeling_act.py, line 354-355
if self.config.conditioning_dim is not None:
    self.conditioning_proj = nn.Embedding(config.conditioning_dim, config.dim_model)
```

- When `conditioning_dim` is set, ACT creates an embedding layer that converts discrete condition labels (0-7) into continuous vectors
- These vectors are injected into the transformer encoder as additional input tokens
- The model learns to process this conditioning signal alongside observations and robot state

### 2. **Transformer Encoder Reception** (line 474-482)

```python
# Task-conditioning token.
if self.config.conditioning_dim is not None:
    if "conditioning" not in batch:
        raise ValueError(
            "`conditioning` must be present in the batch when `conditioning_dim` is enabled."
        )
    conditioning = batch["conditioning"]
    if isinstance(conditioning, Tensor) and conditioning.dim() > 1:
        conditioning = conditioning.squeeze(-1)
    encoder_in_tokens.append(self.conditioning_proj(conditioning.long()))
```

- Looks for `batch["conditioning"]` key (provided by your dataset loader)
- Converts the label to a dense embedding
- Appends it as a token in the encoder input sequence (alongside image features, robot state, etc.)

## Dataset Setup

### Episode-Level vs Frame-Level Conditioning

The conditioning is **episode-level**:
- Each episode (all 100 frames from a single "pick event") gets ONE label (0-7)
- The label is stored in episode metadata, not per-frame
- When you load a frame, the dataset looks up its episode and adds `batch["conditioning"]`

### Your 8×8 Structure

```
Episodes 0-7:   First sequence (picking from position 0-7)
Episodes 8-15:  Second sequence (positions 0-7)
Episodes 16-23: Third sequence (positions 0-7)
... and so on ...
Episodes 56-63: Eighth sequence (positions 0-7)
```

**Conditioning mapping (episode_index % 8):**
- Episode 0 → Class 0 (position 0)
- Episode 1 → Class 1 (position 1)
- Episode 8 → Class 0 (position 0, reset)
- Episode 16 → Class 0 (position 0, reset)

This is already prepared in `conditioning_map_8class.json`.

## Implementation Steps

### STEP 1: Add Conditioning to Dataset

```bash
cd /home/josephrigal/workspace/robotics

# Dry-run (no changes)
PYTHONPATH=./src/lerobot python3 -m lerobot.scripts.lerobot_add_conditioning_labels \
  --repo-id jogarulfo/dataset_MVP_store_cardboard \
  --mapping-json /home/josephrigal/workspace/conditioning_map_8class.json \
  --dry-run

# Actual write (modifies meta/episodes/... parquet files)
PYTHONPATH=./src/lerobot python3 -m lerobot.scripts.lerobot_add_conditioning_labels \
  --repo-id jogarulfo/dataset_MVP_store_cardboard \
  --mapping-json /home/josephrigal/workspace/robotics/conditioning_map_8class.json \
  --overwrite-existing
```

**What this does:**
- Reads each episode's metadata from `meta/episodes/chunk-*/file-*.parquet`
- Adds or updates the `conditioning` column with values from your JSON mapping
- Writes the modified metadata back in-place

### STEP 2: Train ACT with Conditioning Enabled

```bash
cd /home/josephrigal/workspace/robotics

lerobot-train \
  --dataset.repo_id=jogarulfo/dataset_MVP_store_cardboard \
  --policy.type=act \
  --output_dir=outputs/train/act_model_to_store_cardboard \
  --policy.conditioning_dim=8 \
  --batch_size=16 \
  --policy.device=cuda \
  --wandb.enable=false \
  --job_name=act_model_to_store_cardboard \
  --policy.repo_id="jogarulfo/model_to_store_cardboard" \
  --steps=80_000 \
  --save_freq=5_000 
```



**Key parameters:**
- `policy.conditioning_dim 8`: Tells ACT to create an 8-class embedding layer
- Without this, conditioning is disabled (backward compatible)
- The batch loader automatically injects `batch["conditioning"]` from episode metadata

### STEP 3: Validate During Training

The training loop will:
1. Load batches with `batch["conditioning"]` tensors
2. Pass them to ACT's forward method
3. ACT embeds them and adds to the transformer sequence
4. Loss computation includes conditioning-aware predictions

You'll know it's working if:
- Training starts without errors (ACT won't throw "conditioning not in batch" error)
- Loss values make sense (not NaN)
- Validation performance is reasonable

## Debugging

### Check if conditioning is in dataset:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("jogarulfo/dataset_MVP_store_cardboard")
sample = ds[0]
print("conditioning" in sample)  # Should be True
print(sample["conditioning"])    # Should be 0-7 integer
```

### Check if ACT recognizes conditioning_dim:

```python
from lerobot.policies.act.configuration_act import ACTConfig

cfg = ACTConfig(conditioning_dim=8)
print(cfg.conditioning_dim)  # Should be 8
```

### Inspect batch during training:

Add to training script debugging:
```python
batch = next(dl_iter)
if "conditioning" in batch:
    print(f"Conditioning shape: {batch['conditioning'].shape}")
    print(f"Conditioning values: {batch['conditioning'][:5]}")
else:
    print("WARNING: No conditioning in batch!")
```

## Advanced: Custom Conditioning Logic

If you want different conditioning semantics (e.g., color of the box, grip force, etc.):

1. **Modify the mapping**: Change `conditioning_map_8class.json` to map episodes to different classes
2. **Increase classes**: If you want 10 classes instead of 8, set `conditioning_dim 10` and ensure mapping uses 0-9
3. **Multiple signals**: Currently limited to one conditioning signal per episode, but you can modify the architecture to support multiple

## Files in This Setup

```
/home/josephrigal/workspace/
├── conditioning_map_8class.json          # Episode → class mapping
├── test_conditioning.py                  # Validate mapping
├── add_conditioning_to_dataset.py        # Apply mapping (requires pandas)
├── setup_conditioning.sh                 # Full guide (this file)
└── README_conditioning.md                # Technical details (this file)
```

## Next Steps

1. **Install pandas** if not already: `pip install pandas`
2. **Run Step 1** to add conditioning to dataset
3. **Run Step 2** with your training command
4. **Monitor** the first few steps to ensure conditioning is being used
5. **Compare results** with/without conditioning to measure the benefit

---

**Questions?**
- Check [robotics/lerobot/src/lerobot/policies/act/modeling_act.py](robotics/lerobot/src/lerobot/policies/act/modeling_act.py) for the forward pass details
- See [robotics/lerobot/src/lerobot/datasets/lerobot_dataset.py](robotics/lerobot/src/lerobot/datasets/lerobot_dataset.py) lines 1084-1089 for how conditioning is loaded into batches
- Review the test in [robotics/lerobot/tests/datasets/test_datasets.py](robotics/lerobot/tests/datasets/test_datasets.py) line 225 for a simple example
