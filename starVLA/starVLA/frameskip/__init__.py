# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
FrameSkip: Frame-level compression for VLA training.

FrameSkip is a method for reducing redundancy in robot demonstration data
by selectively skipping less important frames during training.

Example usage:
    >>> from starVLA.frameskip import FrameSkipSingleDataset
    >>> dataset = FrameSkipSingleDataset(
    ...     dataset_path="/path/to/data",
    ...     modality_configs=modality_config,
    ...     embodiment_tag="libero_franka",
    ...     frameskip_config={"enabled": True, "compression_ratios": [0.7, 0.8, 0.9, 1.0]},
    ... )
"""

__version__ = "1.0.0"

from starVLA.frameskip.dataloader import (
    FrameSkipSingleDataset,
    FrameSkipMixtureDataset,
    create_frameskip_dataset,
)

from starVLA.frameskip.utils import (
    FrameImportanceCalculator,
    ActionOnlyImportanceCalculator,
    GripperAwareImportanceCalculator,
    FramePruner,
    TemporalConsistentPruner,
    FrameSkipCacheManager,
)

__all__ = [
    # Datasets
    "FrameSkipSingleDataset",
    "FrameSkipMixtureDataset",
    "create_frameskip_dataset",
    # Utils
    "FrameImportanceCalculator",
    "ActionOnlyImportanceCalculator",
    "GripperAwareImportanceCalculator",
    "FramePruner",
    "TemporalConsistentPruner",
    "FrameSkipCacheManager",
    # Training (lazy import to avoid Accelerator initialization at import time)
    # from starVLA.frameskip.training import FrameSkipVLATrainer, build_frameskip_dataloader, frameskip_main
]
