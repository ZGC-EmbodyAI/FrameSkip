# FrameSkip: Learning from Fewer but More Informative Frames in VLA Training

This repository is the anonymous code release for the double-blind review of
**FrameSkip**. FrameSkip is a training-time frame selection framework for
Vision-Language-Action (VLA) policies. It reduces dense robot demonstration
trajectories to fewer, more informative supervision frames while keeping the
policy architecture and inference procedure unchanged.

The implementation is built on top of the starVLA training and evaluation
stack. The FrameSkip-specific code is located in:

```text
starVLA/starVLA/frameskip/
```

The rest of `starVLA/` is included as the base VLA codebase needed to run the
training and evaluation pipeline.

## What FrameSkip Does

FrameSkip changes the data layer rather than the model. For each demonstration
trajectory, it:

1. computes frame-importance scores from lightweight trajectory cues;
2. keeps a target ratio of high-importance frames;
3. remaps dataloader queries to the retained frame indices during training.

At test time, the learned policy is used exactly like the corresponding
starVLA policy. No inference-time frame selector is required.

## Main Components

- `starVLA/starVLA/frameskip/utils/importance_metrics.py`
  implements action-variance, visual-action coherence, task-progress, and
  gripper-aware importance scoring.
- `starVLA/starVLA/frameskip/utils/frame_pruner.py`
  converts importance scores into retained frame indices under configurable
  compression ratios.
- `starVLA/starVLA/frameskip/utils/cache_manager.py`
  stores per-trajectory pruning results so repeated training runs do not need
  to recompute frame scores.
- `starVLA/starVLA/frameskip/dataloader/frameskip_dataset.py`
  extends the LeRobot/starVLA datasets with FrameSkip-aware indexing.
- `starVLA/starVLA/frameskip/training/train_frameskip.py`
  wraps the standard starVLA trainer with FrameSkip dataloader construction,
  dynamic compression-ratio sampling, curriculum options, and logging.

## Repository Layout

```text
.
├── README.md
└── starVLA/
    ├── starVLA/
    │   ├── frameskip/              # FrameSkip implementation
    │   ├── dataloader/             # Base starVLA dataloaders
    │   ├── model/                  # Base VLA model code
    │   └── training/               # Base training code
    ├── examples/                   # Training/evaluation examples
    ├── deployment/                 # Policy server utilities
    └── docs/                       # Base starVLA documentation
```

## Basic Usage

Install the included starVLA package in editable mode:

```bash
cd starVLA
pip install -e .
```

Enable FrameSkip from a starVLA training config by adding a `frameskip` block:

```yaml
frameskip:
  enabled: true
  default_compression_ratio: 0.7
  cache_dir: ./cache/frameskip

  importance:
    type: gripper_aware
    alpha: 0.5
    beta: 0.0
    gamma: 0.2
    enable_vac: false
    gripper_boost: 2.0

  pruning:
    type: temporal_consistent
    compression_ratios: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    used_compression_ratios: [0.7, 0.8, 0.9, 1.0]
    preserve_key_stages: true
    min_frames_per_trajectory: 5
    max_gap: 5

  training:
    dynamic_ratio: true
    ratio_schedule: uniform
    warmup_steps: 0
```

Then launch training with the FrameSkip training entry point:

```bash
cd starVLA
accelerate launch starVLA/frameskip/training/train_frameskip.py \
  --config_file path/to/your_training_config.yaml
```

The first run creates a FrameSkip cache for the configured datasets and
compression ratios. Later runs reuse the cache if the scoring/pruning
configuration is unchanged.

## Checkpoints

For double-blind review, checkpoint links are intentionally omitted from this
README. If checkpoints are provided as part of the review artifact, place them
under a local path such as:

```text
checkpoints/frameskip/
```

FrameSkip checkpoints follow the standard starVLA checkpoint format and can be
loaded by the same policy server and evaluation scripts used for starVLA models.

## Notes for Reviewers

- FrameSkip is a data-selection method for VLA training; it does not introduce
  additional inference modules.
- The default implementation can run with action-only or gripper-aware scoring,
  which avoids visual encoder overhead.
- Visual-action coherence scoring is optional and can use a local timm
  checkpoint when internet access is unavailable.
- Cache generation is rank-0 only in distributed training; other ranks wait for
  the cache to appear before loading it.

## Citation

Citation information is omitted during double-blind review and will be added
after the review period.
