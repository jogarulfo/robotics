"""
ACT Conditioning Layer Study Guide
===================================

This file shows the key code snippets from the ACT model to help you understand
how conditioning works end-to-end.

Reference files:
- robotics/lerobot/src/lerobot/policies/act/modeling_act.py
- robotics/lerobot/src/lerobot/policies/act/configuration_act.py
- robotics/lerobot/src/lerobot/datasets/lerobot_dataset.py
"""

# ============================================================================
# PART 1: ACT Config - Declaration of conditioning_dim
# ============================================================================
# From: configuration_act.py, line 89

@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy."""

    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    # NEW: Conditioning parameter
    conditioning_dim: int | None = None  # Set to 8 for your 8-class setup
    # If None: no conditioning
    # If 8: creates an embedding layer for 8 classes (0-7)


# ============================================================================
# PART 2: ACT Model - Embedding Layer Creation
# ============================================================================
# From: modeling_act.py, lines 354-365

class ACT(nn.Module):
    """Action Chunking Transformer with optional conditioning."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # ... other initialization ...

        # Creates the conditioning embedding if enabled
        if self.config.conditioning_dim is not None:
            # Maps class labels (0-7) to continuous vectors (dim_model=512 by default)
            # This is learned during training
            self.conditioning_proj = nn.Embedding(config.conditioning_dim, config.dim_model)
            print(f"Initialized conditioning embedding: {config.conditioning_dim} classes → {config.dim_model} dims")

        # Compute how many 1D tokens we need in encoder
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        if self.config.conditioning_dim is not None:
            n_1d_tokens += 1  # Add 1 for conditioning token!

        # Positional embeddings for all 1D tokens (robot state, env state, conditioning)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)


# ============================================================================
# PART 3: ACT Forward Pass - Using Conditioning
# ============================================================================
# From: modeling_act.py, lines 474-482

def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """
    Args:
        batch: Dictionary containing observations, actions, and optionally 'conditioning'
               Each key in batch is replicated across all frames of episodes
    """

    # ... setup code ...

    # Prepare transformer encoder inputs
    encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

    # Robot state token (if available)
    if self.config.robot_state_feature:
        encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

    # Environment state token (if available)
    if self.config.env_state_feature:
        encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

    # *** CONDITIONING TOKEN (NEW) ***
    if self.config.conditioning_dim is not None:
        # 1. Check that conditioning is in the batch
        if "conditioning" not in batch:
            raise ValueError(
                "`conditioning` must be present in the batch when `conditioning_dim` is enabled."
            )

        # 2. Get the conditioning data from batch
        conditioning = batch["conditioning"]  # Shape: (batch_size,) with values 0-7

        # 3. Handle edge case: squeeze if it's 2D
        if isinstance(conditioning, Tensor) and conditioning.dim() > 1:
            conditioning = conditioning.squeeze(-1)

        # 4. Embed the conditioning class labels
        # nn.Embedding expects LongTensor, so convert: (batch_size,) → (batch_size, dim_model)
        conditioning_embed = self.conditioning_proj(conditioning.long())

        # 5. Append the conditioning token to encoder inputs
        encoder_in_tokens.append(conditioning_embed)

    # ... rest of forward pass ...


# ============================================================================
# PART 4: Dataset - Loading Conditioning into Batch
# ============================================================================
# From: lerobot_dataset.py, lines 1084-1089

class LeRobotDataset:
    """Loads frames and adds episode-level metadata like conditioning."""

    def __getitem__(self, idx: int) -> dict:
        """Get one frame."""

        # ... load observation data, actions, etc. ...

        # Check if this episode has a conditioning label
        ep_idx = ...  # episode index for this frame
        if ep_idx in self._episode_conditioning:
            # Episode-level conditioning stored in memory
            item["conditioning"] = self._episode_conditioning[ep_idx]
        elif self.meta.episodes is not None and ep_idx < len(self.meta.episodes):
            # Or check episode metadata from parquet
            episode_meta = self.meta.episodes[ep_idx]
            if "conditioning" in episode_meta and episode_meta["conditioning"] is not None:
                item["conditioning"] = episode_meta["conditioning"]

        return item  # Now includes "conditioning" key!


# ============================================================================
# PART 5: Training Loop - Batching
# ============================================================================
# From: lerobot_train.py (conceptual)

# During training:
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for batch in dataloader:
    # batch is now:
    # {
    #   "observation.image": (32, 3, 224, 224),
    #   "observation.state": (32, 7),
    #   "action": (32, 100, 4),
    #   "conditioning": (32,) with values [0-7, 0-7, ...]  ← NEW!
    # }

    # Pass to policy (ACT)
    loss, loss_dict = policy(batch)

    # ACT forward():
    #   1. Sees batch["conditioning"] exists
    #   2. Embeds it: (32,) → (32, 512)
    #   3. Adds as token in transformer sequence
    #   4. Learns position-aware representations


# ============================================================================
# PART 6: Verification - Check Your Setup
# ============================================================================

def verify_conditioning_setup():
    """Check that everything is wired correctly."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    import torch

    # 1. Check dataset has conditioning
    ds = LeRobotDataset("jogarulfo/dataset_MVP_store_cardboard")
    sample = ds[0]
    assert "conditioning" in sample, "conditioning not found in dataset sample!"
    assert isinstance(sample["conditioning"], (int, float, torch.Tensor)), \
        f"conditioning should be numeric, got {type(sample['conditioning'])}"
    print(f"✓ Dataset conditioning: sample = {sample['conditioning']} (type: {type(sample['conditioning']).__name__})")

    # 2. Check ACT config can be created with conditioning_dim
    cfg = ACTConfig(conditioning_dim=8)
    assert cfg.conditioning_dim == 8, "conditioning_dim not set correctly!"
    print(f"✓ ACT config: conditioning_dim = {cfg.conditioning_dim}")

    # 3. Check embedding layer exists
    from lerobot.policies.act.modeling_act import ACT
    model = ACT(cfg)
    assert hasattr(model, 'conditioning_proj'), "conditioning_proj not found!"
    print(f"✓ ACT model: conditioning_proj = {model.conditioning_proj}")

    # 4. Load a batch and check conditioning
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4)
    batch = next(iter(dl))
    assert "conditioning" in batch, "conditioning not in batch!"
    print(f"✓ DataLoader batch: conditioning shape = {batch['conditioning'].shape}")
    print(f"  Values: {batch['conditioning']}")

    print("\n✓✓✓ All conditioning components verified! ✓✓✓")

if __name__ == "__main__":
    verify_conditioning_setup()
