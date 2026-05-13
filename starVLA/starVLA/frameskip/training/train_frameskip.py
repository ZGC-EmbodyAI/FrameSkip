# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
FrameSkip VLA Training Script.

Extends the standard VLA training with FrameSkip support.
This script allows training VLA models with frame-level compression.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from starVLA.training.train_starvla import (
    VLATrainer,
    build_model,
    setup_directories,
    setup_optimizer_and_scheduler,
    normalize_dotlist_args,
    accelerator,  # reuse the module-level accelerator from train_starvla
)
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.dataloader import build_dataloader as original_build_dataloader

from starVLA.frameskip.dataloader import FrameSkipSingleDataset, FrameSkipMixtureDataset
from starVLA.frameskip.utils import FrameImportanceCalculator, FramePruner

import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb


logger = get_logger(__name__)


class FrameSkipVLATrainer(VLATrainer):
    """
    VLA Trainer with FrameSkip support.
    
    Extends the base VLATrainer to support:
    - Dynamic compression ratio sampling during training
    - Curriculum learning for compression ratios
    - FrameSkip-specific logging
    """
    
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        super().__init__(cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator)
        
        # FrameSkip configuration
        self.frameskip_config = cfg.get("frameskip", {})
        self.use_frameskip = self.frameskip_config.get("enabled", False)
        
        if self.use_frameskip:
            pruning_cfg = self.frameskip_config.get("pruning", {})
            self.compression_ratios = pruning_cfg.get(
                "used_compression_ratios",
                pruning_cfg.get("compression_ratios", [0.7, 0.8, 0.9, 1.0]),
            )
            self.dynamic_ratio = self.frameskip_config.get(
                "training", {}).get("dynamic_ratio", True)
            self.ratio_schedule = self.frameskip_config.get(
                "training", {}).get("ratio_schedule", "uniform")
            
            # For curriculum learning
            self.warmup_steps = self.frameskip_config.get(
                "training", {}).get("warmup_steps", 0)
    
    def _get_next_batch(self):
        """Get next batch with dynamic compression ratio."""
        if self.use_frameskip and self.dynamic_ratio:
            # Determine ratio before fetching batch so dataset uses it during __getitem__
            if self.completed_steps < self.warmup_steps:
                ratio = 1.0
            else:
                ratio = self._sample_compression_ratio()

            # Set ratio on all FrameSkipSingleDataset instances in the dataloader
            dataset = self.vla_train_dataloader.dataset
            if hasattr(dataset, "set_compression_ratio"):
                dataset.set_compression_ratio(ratio)
            elif hasattr(dataset, "datasets"):
                for ds in dataset.datasets:
                    if hasattr(ds, "set_compression_ratio"):
                        ds.set_compression_ratio(ratio)

            if self.completed_steps % 100 == 0:
                logger.info(f"Step {self.completed_steps}: Using compression ratio {ratio:.2f}")

        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla
    
    def _sample_compression_ratio(self) -> float:
        """Sample compression ratio based on schedule."""
        if self.ratio_schedule == "uniform":
            # Uniform random sampling
            return float(np.random.choice(self.compression_ratios))
        
        elif self.ratio_schedule == "curriculum":
            # Curriculum learning: start with high ratios, gradually decrease
            progress = min(1.0, (self.completed_steps - self.warmup_steps) / 
                          (self.config.trainer.max_train_steps - self.warmup_steps))
            
            # Bias towards higher ratios early, lower ratios later
            sorted_ratios = sorted(self.compression_ratios, reverse=True)
            idx = int(progress * (len(sorted_ratios) - 1))
            return float(sorted_ratios[min(idx, len(sorted_ratios) - 1)])
        
        elif self.ratio_schedule == "adaptive":
            # Adaptive: sample based on recent performance
            # This would require tracking recent metrics
            return float(np.random.choice(self.compression_ratios))
        
        else:
            return float(np.random.choice(self.compression_ratios))
    
    def _train_step(self, batch_vla, batch_vlm=None):
        """Training step with FrameSkip logging."""
        # Get compression ratio for logging
        compression_ratios = [s.get("compression_ratio", 1.0) for s in batch_vla]
        avg_ratio = np.mean(compression_ratios)
        
        # Run parent training step
        step_metrics = super()._train_step(batch_vla, batch_vlm)
        
        # Add FrameSkip metrics
        if self.use_frameskip:
            step_metrics["compression_ratio"] = avg_ratio
            step_metrics["effective_frames"] = avg_ratio * len(batch_vla)
        
        return step_metrics
    
    def _log_metrics(self, metrics):
        """Log metrics with FrameSkip info."""
        if self.use_frameskip and "compression_ratio" in metrics:
            if dist.get_rank() == 0:
                wandb.log({
                    "frameskip/compression_ratio": metrics["compression_ratio"],
                    "frameskip/effective_frames": metrics.get("effective_frames", 0),
                }, step=self.completed_steps)
        
        super()._log_metrics(metrics)


def build_frameskip_dataloader(cfg, accelerator, output_dir):
    """
    Build dataloader with FrameSkip support.

    This function extends the original build_dataloader to support FrameSkip.
    """
    frameskip_cfg = cfg.get("frameskip", {})

    if not frameskip_cfg.get("enabled", False):
        # FrameSkip disabled, use original loader
        return original_build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    logger.info("Building FrameSkip dataloader...")

    from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
    from starVLA.dataloader.lerobot_datasets import collate_fn
    from torch.utils.data import DataLoader

    data_cfg = cfg.datasets.vla_data
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    video_backend = data_cfg.get("video_backend", "decord")

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, dataset_mixture = set(), []

    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            logger.info(f"Skipping duplicate dataset: {dataset_key}")
            continue
        included_datasets.add(dataset_key)

        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        modality_config = data_config.modality_config()
        transforms = data_config.transform()
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type, EmbodimentTag.NEW_EMBODIMENT)

        dataset = FrameSkipSingleDataset(
            dataset_path=data_root_dir / d_name,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            frameskip_config=frameskip_cfg,
        )
        dataset_mixture.append((dataset, d_weight))

    train_dataset = LeRobotMixtureDataset(
        dataset_mixture,
        mode="train",
        balance_dataset_weights=data_cfg.get("balance_dataset_weights", True),
        balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", True),
        seed=cfg.seed,
        data_cfg=data_cfg,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir = Path(cfg.output_dir)
        train_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    num_workers = max(4, 16 // num_gpus)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg.per_device_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    logger.info(f"FrameSkip dataloader created: {len(train_dataset)} samples")
    return train_dataloader


def frameskip_main(cfg):
    """Main training function with FrameSkip."""
    logger.info("FrameSkip VLA Training :: Warming Up")

    # Wrap config
    cfg = wrap_config(cfg)
    logger.info("Configuration wrapped for access tracking")

    # Setup directories
    output_dir = setup_directories(cfg)

    # Build model (reuse existing logic)
    model = build_model(cfg)

    # Build FrameSkip dataloader
    vla_train_dataloader = build_frameskip_dataloader(cfg, accelerator, output_dir)
    
    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    
    # Create trainer
    trainer = FrameSkipVLATrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    
    # Prepare and train
    trainer.prepare_training()
    trainer.train()
    
    logger.info("FrameSkip Training Complete!")
    
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_frameskip.yaml",
        help="Path to YAML config"
    )
    args, clipargs = parser.parse_known_args()
    
    # Load config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    # Debug mode
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    
    frameskip_main(cfg)
