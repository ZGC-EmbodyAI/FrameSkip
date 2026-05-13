# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Frame Pruner for FrameSkip.

This module implements the frame pruning logic based on importance scores
and compression ratios.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class PruningResult:
    """Result of frame pruning."""
    keep_indices: np.ndarray
    compressed_trajectory: Dict[str, Any]
    compression_ratio: float
    actual_ratio: float
    importance_scores: np.ndarray


class FramePruner:
    """
    Prunes frames from trajectories based on importance scores.
    
    Implements the TokenSkip-style pruning:
    1. Compute importance threshold based on compression ratio
    2. Keep frames above threshold
    3. Ensure minimum number of frames
    4. Preserve key stages if configured
    """
    
    def __init__(
        self,
        compression_ratios: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        preserve_key_stages: bool = True,
        min_frames_per_trajectory: int = 5,
        key_stage_boost: float = 1.5,
    ):
        """
        Initialize the frame pruner.
        
        Args:
            compression_ratios: List of compression ratios to support
            preserve_key_stages: Whether to preserve frames at key stages
            min_frames_per_trajectory: Minimum frames to keep per trajectory
            key_stage_boost: Boost factor for key stage frames
        """
        self.compression_ratios = sorted(compression_ratios)
        self.preserve_key_stages = preserve_key_stages
        self.min_frames_per_trajectory = min_frames_per_trajectory
        self.key_stage_boost = key_stage_boost
    
    def prune_trajectory(
        self,
        trajectory: Dict[str, Any],
        importance_scores: np.ndarray,
        compression_ratio: float,
    ) -> PruningResult:
        """
        Prune a trajectory based on importance scores.
        
        Args:
            trajectory: Trajectory dictionary containing observations, actions, etc.
            importance_scores: Importance score for each frame
            compression_ratio: Target compression ratio (gamma)
            
        Returns:
            PruningResult containing keep indices and compressed trajectory
        """
        num_frames = len(importance_scores)
        
        if num_frames == 0:
            return PruningResult(
                keep_indices=np.array([]),
                compressed_trajectory={},
                compression_ratio=compression_ratio,
                actual_ratio=0.0,
                importance_scores=importance_scores,
            )
        
        # Handle full retention
        if compression_ratio >= 1.0:
            return PruningResult(
                keep_indices=np.arange(num_frames),
                compressed_trajectory=trajectory.copy(),
                compression_ratio=1.0,
                actual_ratio=1.0,
                importance_scores=importance_scores,
            )
        
        # Calculate target number of frames
        target_keep = max(self.min_frames_per_trajectory, int(num_frames * compression_ratio))
        
        # Compute threshold based on compression ratio
        # Use quantile to find the importance threshold
        threshold = np.quantile(importance_scores, 1.0 - compression_ratio)
        
        # Initial mask: keep frames above threshold
        keep_mask = importance_scores >= threshold
        
        # Preserve key stages if configured
        if self.preserve_key_stages:
            key_stage_indices = self._detect_key_stages(trajectory, importance_scores)
            keep_mask[key_stage_indices] = True
        
        # Get current keep indices
        keep_indices = np.where(keep_mask)[0]
        
        # Adjust to meet target count
        if len(keep_indices) > target_keep:
            # Too many frames, select top by importance
            keep_indices = self._select_top_frames(
                importance_scores, keep_indices, target_keep
            )
        elif len(keep_indices) < target_keep:
            # Too few frames, add more high-importance frames
            keep_indices = self._add_frames_to_meet_target(
                importance_scores, keep_indices, target_keep, num_frames
            )
        
        # Ensure sorted order
        keep_indices = np.sort(keep_indices)
        
        # Create compressed trajectory
        compressed_trajectory = self._create_compressed_trajectory(
            trajectory, keep_indices
        )
        
        actual_ratio = len(keep_indices) / num_frames
        
        return PruningResult(
            keep_indices=keep_indices,
            compressed_trajectory=compressed_trajectory,
            compression_ratio=compression_ratio,
            actual_ratio=actual_ratio,
            importance_scores=importance_scores[keep_indices],
        )
    
    def create_multi_ratio_trajectory(
        self,
        trajectory: Dict[str, Any],
        importance_scores: np.ndarray,
    ) -> Dict[float, PruningResult]:
        """
        Create multiple compressed versions of a trajectory.
        
        Args:
            trajectory: Original trajectory
            importance_scores: Importance scores for each frame
            
        Returns:
            Dictionary mapping compression ratios to PruningResults
        """
        results = {}
        
        for ratio in self.compression_ratios:
            result = self.prune_trajectory(trajectory, importance_scores, ratio)
            results[ratio] = result
        
        return results
    
    def _detect_key_stages(
        self,
        trajectory: Dict[str, Any],
        importance_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Detect key stages in a trajectory.
        
        Key stages include:
        1. Gripper state changes
        2. Large action changes
        3. First and last frames
        
        Args:
            trajectory: Trajectory dictionary
            importance_scores: Importance scores
            
        Returns:
            Array of key stage indices
        """
        num_frames = len(importance_scores)
        key_indices = set()
        
        # Always keep first and last frames
        key_indices.add(0)
        key_indices.add(num_frames - 1)
        
        actions = trajectory.get("actions", np.array([]))
        
        if len(actions) > 1:
            # Detect gripper state changes
            gripper_changes = self._detect_gripper_changes(actions)
            key_indices.update(gripper_changes)
            
            # Detect large action changes (above 90th percentile)
            action_diff = np.diff(actions, axis=0)
            action_magnitudes = np.linalg.norm(action_diff, axis=1)
            threshold = np.percentile(action_magnitudes, 90)
            large_changes = np.where(action_magnitudes >= threshold)[0] + 1
            key_indices.update(large_changes)
        
        return np.array(list(key_indices))
    
    def _detect_gripper_changes(self, actions: np.ndarray) -> np.ndarray:
        """Detect indices where gripper state changes."""
        if actions.shape[1] < 7:
            return np.array([])
        
        # Assume last dimension is gripper
        gripper = actions[:, -1]
        
        # Detect changes
        changes = np.abs(np.diff(gripper, prepend=gripper[0]))
        change_indices = np.where(changes > 0.1)[0]  # Threshold for gripper change
        
        return change_indices
    
    def _select_top_frames(
        self,
        importance_scores: np.ndarray,
        candidate_indices: np.ndarray,
        target_count: int,
    ) -> np.ndarray:
        """Select top frames by importance from candidates."""
        if len(candidate_indices) <= target_count:
            return candidate_indices
        
        # Get importance scores for candidates
        candidate_scores = importance_scores[candidate_indices]
        
        # Sort by importance (descending)
        sorted_idx = np.argsort(candidate_scores)[::-1]
        
        # Select top target_count
        selected = candidate_indices[sorted_idx[:target_count]]
        
        return np.sort(selected)
    
    def _add_frames_to_meet_target(
        self,
        importance_scores: np.ndarray,
        current_indices: np.ndarray,
        target_count: int,
        total_frames: int,
    ) -> np.ndarray:
        """Add high-importance frames to meet target count."""
        if len(current_indices) >= target_count:
            return current_indices
        
        # Get available frames
        all_indices = np.arange(total_frames)
        available = np.setdiff1d(all_indices, current_indices)
        
        if len(available) == 0:
            return current_indices
        
        # Get importance of available frames
        available_scores = importance_scores[available]
        
        # Sort by importance (descending)
        sorted_idx = np.argsort(available_scores)[::-1]
        
        # Select top frames to meet target
        num_needed = min(target_count - len(current_indices), len(available))
        additional = available[sorted_idx[:num_needed]]
        
        # Combine and sort
        return np.sort(np.concatenate([current_indices, additional]))
    
    def _create_compressed_trajectory(
        self,
        trajectory: Dict[str, Any],
        keep_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Create a compressed version of the trajectory.
        
        Args:
            trajectory: Original trajectory
            keep_indices: Indices to keep
            
        Returns:
            Compressed trajectory dictionary
        """
        compressed = {}
        
        for key, value in trajectory.items():
            if key == "observations":
                # Handle list of frames
                if isinstance(value, list):
                    if len(value) == 0:
                        compressed[key] = []
                    else:
                        compressed[key] = [value[i] for i in keep_indices if i < len(value)]
                else:
                    compressed[key] = value[keep_indices] if hasattr(value, '__getitem__') else value
            elif key == "actions":
                # Handle numpy array actions
                compressed[key] = value[keep_indices]
            elif key == "states":
                # Handle states if present
                compressed[key] = value[keep_indices]
            elif key == "rewards":
                # Handle rewards if present
                compressed[key] = value[keep_indices]
            else:
                # Keep other metadata as is
                compressed[key] = value
        
        # Add metadata about compression
        compressed["_frameskip_metadata"] = {
            "keep_indices": keep_indices,
            "original_length": len(trajectory.get("actions", [])),
            "compressed_length": len(keep_indices),
        }
        
        return compressed


class TemporalConsistentPruner(FramePruner):
    """
    Pruner that maintains temporal consistency.
    
    Ensures that kept frames are temporally coherent and not too sparse.
    """
    
    def __init__(
        self,
        max_gap: int = 5,
        **kwargs,
    ):
        """
        Initialize temporal consistent pruner.
        
        Args:
            max_gap: Maximum allowed gap between consecutive kept frames
            **kwargs: Additional arguments for base FramePruner
        """
        super().__init__(**kwargs)
        self.max_gap = max_gap
    
    def prune_trajectory(
        self,
        trajectory: Dict[str, Any],
        importance_scores: np.ndarray,
        compression_ratio: float,
    ) -> PruningResult:
        """Prune with temporal consistency constraint."""
        # Get base pruning result
        result = super().prune_trajectory(trajectory, importance_scores, compression_ratio)
        
        if len(result.keep_indices) == 0:
            return result
        
        # Fill in gaps if too large
        keep_indices = result.keep_indices
        filled_indices = self._fill_gaps(keep_indices, len(importance_scores))
        
        if len(filled_indices) > len(keep_indices):
            # Recreate compressed trajectory with filled indices
            compressed_trajectory = self._create_compressed_trajectory(
                trajectory, filled_indices
            )
            actual_ratio = len(filled_indices) / len(importance_scores)
            
            return PruningResult(
                keep_indices=filled_indices,
                compressed_trajectory=compressed_trajectory,
                compression_ratio=compression_ratio,
                actual_ratio=actual_ratio,
                importance_scores=importance_scores[filled_indices],
            )
        
        return result
    
    def _fill_gaps(
        self,
        keep_indices: np.ndarray,
        total_frames: int,
    ) -> np.ndarray:
        """Fill in large gaps between kept frames."""
        if len(keep_indices) <= 1:
            return keep_indices
        
        filled = [keep_indices[0]]
        
        for i in range(1, len(keep_indices)):
            gap = keep_indices[i] - keep_indices[i - 1]
            
            if gap > self.max_gap:
                # Add intermediate frames
                num_add = gap // self.max_gap
                for j in range(1, num_add + 1):
                    filled.append(keep_indices[i - 1] + j * self.max_gap)
            
            filled.append(keep_indices[i])
        
        return np.array(sorted(filled))


def get_frame_pruner(config: Dict) -> FramePruner:
    """
    Factory function to create frame pruner from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FramePruner instance
    """
    # Make a copy to avoid modifying the original config
    config = config.copy()
    pruner_type = config.pop("type", "default")
    config.pop("used_compression_ratios", None)
    
    if pruner_type == "temporal_consistent":
        # TemporalConsistentPruner accepts max_gap
        return TemporalConsistentPruner(**config)
    else:
        # Filter out parameters that FramePruner doesn't accept
        frame_pruner_params = {
            "compression_ratios", "preserve_key_stages",
            "min_frames_per_trajectory", "key_stage_boost"
        }
        filtered_config = {k: v for k, v in config.items() if k in frame_pruner_params}
        return FramePruner(**filtered_config)
