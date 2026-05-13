# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Frame Importance Metrics for FrameSkip.

This module implements multiple frame importance calculation strategies:
1. AVI (Action Variance Importance): Based on action changes between frames
2. VAC (Visual-Action Coherence): Based on visual and action consistency
3. TPI (Task Progress Importance): Based on task completion progress
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image


class FrameImportanceCalculator:
    """
    Calculator for frame importance scores.
    
    Combines multiple importance metrics:
    - AVI (Action Variance Importance): Measures action changes
    - VAC (Visual-Action Coherence): Measures visual-action alignment
    - TPI (Task Progress Importance): Measures task progress contribution
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        enable_vac: bool = False,
        vac_beta: Optional[float] = None,
        use_visual_encoder: Optional[bool] = None,  # Deprecated, kept for backward compatibility
        visual_encoder_name: Optional[str] = None,
        visual_encoder_checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the importance calculator.
        
        Args:
            alpha: Weight for AVI (Action Variance Importance)
            beta: Weight for VAC (Visual-Action Coherence)
            gamma: Weight for TPI (Task Progress Importance)
            enable_vac: Whether to enable VAC (Visual-Action Coherence) metric
            vac_beta: VAC weight when enable_vac is True. If None, uses beta value
            use_visual_encoder: (Deprecated) Use enable_vac instead
            visual_encoder_name: Name of visual encoder (e.g., "dinov2_vits14")
            visual_encoder_checkpoint: Optional local checkpoint path for offline use
            device: Device for computation
        """
        # Handle backward compatibility: use_visual_encoder -> enable_vac
        if use_visual_encoder is not None:
            enable_vac = use_visual_encoder
            
        self.alpha = alpha
        # If enable_vac is True and vac_beta is specified, use it; otherwise use beta
        self.beta = vac_beta if (enable_vac and vac_beta is not None) else beta
        self.gamma = gamma
        self.enable_vac = enable_vac
        self.use_visual_encoder = enable_vac  # Internal flag for compatibility
        self.device = device
        self.visual_encoder_checkpoint = visual_encoder_checkpoint
        
        self.visual_encoder = None
        if enable_vac and visual_encoder_name:
            self.visual_encoder = self._load_visual_encoder(
                visual_encoder_name, visual_encoder_checkpoint
            )
    
    def compute_avi(
        self, 
        actions: np.ndarray,
        window_size: int = 3,
    ) -> np.ndarray:
        """
        Compute Action Variance Importance (AVI).
        
        Measures how much the action changes at each frame.
        Higher values indicate more important frames (key transitions).
        
        Args:
            actions: Action array of shape (T, action_dim)
            window_size: Window size for computing local variance
            
        Returns:
            Importance scores of shape (T,)
        """
        if len(actions) < 2:
            return np.ones(len(actions))
        
        # Compute immediate action difference
        action_diff = np.diff(actions, axis=0)
        avi = np.linalg.norm(action_diff, axis=1)
        
        # Add local variance component (future actions)
        local_var = np.zeros(len(actions) - 1)
        for i in range(len(actions) - 1):
            end_idx = min(i + 1 + window_size, len(actions))
            if end_idx > i + 1:
                local_var[i] = np.var(actions[i+1:end_idx], axis=0).mean()
        
        avi = avi + 0.1 * local_var
        
        # Pad first frame with the value of second frame
        avi = np.concatenate([[avi[0] if len(avi) > 0 else 0], avi])
        
        return avi
    
    def compute_vac(
        self,
        frames: List[Image.Image],
        actions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Visual-Action Coherence (VAC).
        
        Measures the coherence between visual changes and action changes.
        High coherence indicates important frames where action causes visual change.
        
        Args:
            frames: List of PIL Images (T frames)
            actions: Action array of shape (T, action_dim)
            
        Returns:
            Importance scores of shape (T,)
        """
        num_actions = len(actions)

        if len(frames) < 2 or num_actions < 2 or not self.use_visual_encoder or self.visual_encoder is None:
            # Return uniform scores if no visual encoder
            return np.ones(num_actions)
        
        # Extract visual features
        visual_features = self._extract_visual_features(frames)
        
        # Compute visual differences
        visual_diff = np.diff(visual_features, axis=0)
        visual_diff_norm = np.linalg.norm(visual_diff, axis=1)
        
        if len(frames) != num_actions:
            sample_indices = np.linspace(0, num_actions - 1, len(frames), dtype=int)
            sampled_actions = actions[sample_indices]
        else:
            sampled_actions = actions

        # Compute action differences
        action_diff = np.diff(sampled_actions, axis=0)
        action_diff_norm = np.linalg.norm(action_diff, axis=1) + 1e-8
        
        # Compute coherence: visual change / action change
        # High coherence: visual change matches action change (important)
        # Low coherence: visual change without action or vice versa (less important)
        vac = visual_diff_norm / action_diff_norm
        
        # Clip extreme values
        vac = np.clip(vac, 0, np.percentile(vac, 95))
        
        # Pad first frame
        vac = np.concatenate([[vac[0] if len(vac) > 0 else 0], vac])

        if len(vac) != num_actions:
            src_positions = np.linspace(0, num_actions - 1, len(vac))
            dst_positions = np.arange(num_actions)
            vac = np.interp(dst_positions, src_positions, vac)
        
        return vac
    
    def compute_tpi(
        self,
        num_frames: int,
        trajectory_metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Compute Task Progress Importance (TPI).
        
        Heuristic-based importance based on task progress.
        Currently uses a simple Gaussian centered at middle of trajectory.
        Can be extended to use VLM for key stage detection.
        
        Args:
            num_frames: Number of frames in trajectory
            trajectory_metadata: Optional metadata about the trajectory
            
        Returns:
            Importance scores of shape (num_frames,)
        """
        if num_frames == 0:
            return np.array([])
        
        # Simple heuristic: frames in the middle are often more important
        # (approaching goal, critical interactions)
        progress = np.arange(num_frames) / max(num_frames - 1, 1)
        
        # Gaussian weighting centered at 0.5 (middle)
        tpi = np.exp(-((progress - 0.5) ** 2) / 0.2)
        
        # Can be enhanced with key stage detection from metadata
        if trajectory_metadata and "key_stages" in trajectory_metadata:
            key_stages = trajectory_metadata["key_stages"]
            for stage_idx in key_stages:
                if 0 <= stage_idx < num_frames:
                    tpi[stage_idx] *= 1.5  # Boost key stages
        
        return tpi
    
    def compute_importance(
        self,
        trajectory: Dict,
        trajectory_metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Compute combined frame importance scores.
        
        Combines AVI, VAC, and TPI with learned weights.
        
        Args:
            trajectory: Dictionary containing:
                - "observations": List of PIL Images or frame paths
                - "actions": np.ndarray of actions
            trajectory_metadata: Optional metadata
            
        Returns:
            Combined importance scores of shape (T,)
        """
        frames = trajectory.get("observations", [])
        actions = trajectory.get("actions", np.array([]))
        
        if len(actions) == 0:
            return np.array([])
        
        num_frames = len(actions)
        
        # Compute individual metrics
        avi = self.compute_avi(actions)
        
        if self.use_visual_encoder and len(frames) > 0:
            vac = self.compute_vac(frames, actions)
        else:
            vac = np.ones(num_frames)
        
        tpi = self.compute_tpi(num_frames, trajectory_metadata)
        
        # Normalize each metric
        avi_norm = self._normalize(avi)
        vac_norm = self._normalize(vac)
        tpi_norm = self._normalize(tpi)
        
        # Weighted combination
        importance = (
            self.alpha * avi_norm +
            self.beta * vac_norm +
            self.gamma * tpi_norm
        )
        
        # Ensure non-negative
        importance = np.maximum(importance, 0)
        
        return importance
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        if len(scores) == 0:
            return scores
        
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val < 1e-6:
            return np.ones_like(scores)
        
        return (scores - min_val) / (max_val - min_val)
    
    def _load_visual_encoder(
        self,
        encoder_name: str,
        checkpoint_path: Optional[str] = None,
    ):
        """Load visual encoder model."""
        try:
            import timm
            
            checkpoint_kwargs = {}
            pretrained = True
            if checkpoint_path:
                resolved_path = Path(checkpoint_path).expanduser()
                if not resolved_path.exists():
                    print(
                        f"Warning: visual encoder checkpoint {resolved_path} not found; "
                        "falling back to timm pretrained weights (may require internet)."
                    )
                else:
                    checkpoint_kwargs["checkpoint_path"] = str(resolved_path)
                    pretrained = False
            
            model = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                num_classes=0,
                **checkpoint_kwargs,
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: Failed to load visual encoder {encoder_name}: {e}")
            return None
    
    def _extract_visual_features(self, frames: List[Image.Image]) -> np.ndarray:
        """Extract visual features from frames using the encoder."""
        if self.visual_encoder is None:
            return np.zeros((len(frames), 1))
        
        from torchvision import transforms
        
        # Preprocess frames
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        features = []
        with torch.no_grad():
            for frame in frames:
                img_tensor = preprocess(frame).unsqueeze(0).to(self.device)
                feat = self.visual_encoder(img_tensor)
                features.append(feat.cpu().numpy().flatten())
        
        return np.array(features)


class ActionOnlyImportanceCalculator(FrameImportanceCalculator):
    """
    Simplified calculator using only action-based importance (AVI).
    
    This is the default and recommended calculator as it doesn't require
    visual encoding and is computationally efficient.
    """
    
    def __init__(self, **kwargs):
        # Force AVI only
        kwargs['alpha'] = 1.0
        kwargs['beta'] = 0.0
        kwargs['gamma'] = 0.0
        kwargs['use_visual_encoder'] = False
        super().__init__(**kwargs)
    
    def compute_importance(
        self,
        trajectory: Dict,
        trajectory_metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """Compute importance using only AVI."""
        actions = trajectory.get("actions", np.array([]))
        if len(actions) == 0:
            return np.array([])
        return self._normalize(self.compute_avi(actions))


class GripperAwareImportanceCalculator(FrameImportanceCalculator):
    """
    Importance calculator that is aware of gripper state changes.
    
    Gives extra importance to frames where gripper state changes,
    as these are typically critical interaction points.
    """
    
    def __init__(self, gripper_boost: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.gripper_boost = gripper_boost
    
    def compute_importance(
        self,
        trajectory: Dict,
        trajectory_metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """Compute importance with gripper state awareness."""
        # Get base importance
        importance = super().compute_importance(trajectory, trajectory_metadata)
        
        # Detect gripper state changes
        actions = trajectory.get("actions", np.array([]))
        if len(actions) > 0 and actions.shape[1] >= 7:
            # Assume last dimension is gripper
            gripper = actions[:, -1]
            gripper_changes = np.abs(np.diff(gripper, prepend=gripper[0]))
            
            # Boost importance at gripper changes
            importance = importance * (1 + self.gripper_boost * gripper_changes)
        
        return self._normalize(importance)


def get_importance_calculator(config: Dict) -> FrameImportanceCalculator:
    """
    Factory function to create importance calculator from config.
    
    Args:
        config: Configuration dictionary containing:
            - "type": "default", "action_only", "gripper_aware"
            - Other parameters for the calculator
            
    Returns:
        FrameImportanceCalculator instance
    """
    # Make a copy to avoid modifying the original config
    config = config.copy()
    calc_type = config.pop("type", "default")
    
    if calc_type == "action_only":
        return ActionOnlyImportanceCalculator(**config)
    elif calc_type == "gripper_aware":
        return GripperAwareImportanceCalculator(**config)
    else:
        return FrameImportanceCalculator(**config)
