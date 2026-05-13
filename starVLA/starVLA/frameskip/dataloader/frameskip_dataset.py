# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
FrameSkip Dataset Classes.

Extends LeRobot datasets with FrameSkip functionality.
"""

import numpy as np
import pickle
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sequence
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    LeRobotMixtureDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform

from starVLA.frameskip.utils import (
    FrameImportanceCalculator,
    FramePruner,
    FrameSkipCacheManager,
    get_importance_calculator,
    get_frame_pruner,
)


class FrameSkipSingleDataset(LeRobotSingleDataset):
    """
    LeRobotSingleDataset with FrameSkip support.
    
    Extends the base dataset to support frame skipping based on importance scores.
    """
    
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: Dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: Optional[Dict] = None,
        transforms: Optional[ComposedModalityTransform] = None,
        delete_pause_frame: bool = False,
        data_cfg: Optional[Dict] = None,
        frameskip_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize FrameSkip dataset.
        
        Args:
            dataset_path: Path to the dataset
            modality_configs: Modality configurations
            embodiment_tag: Embodiment tag
            video_backend: Video backend ("decord" or "torchvision")
            video_backend_kwargs: Additional kwargs for video backend
            transforms: Composed transforms
            delete_pause_frame: Whether to delete pause frames
            data_cfg: Data configuration
            frameskip_config: FrameSkip configuration
            **kwargs: Additional arguments passed to parent
        """
        # Initialize parent first
        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            **kwargs,
        )
        
        # FrameSkip configuration
        self.frameskip_config = frameskip_config or {}
        self.enable_frameskip = self.frameskip_config.get("enabled", False)
        self.default_compression_ratio = self.frameskip_config.get(
            "default_compression_ratio", 1.0
        )
        self._compression_ratio_tensor = torch.tensor(
            [float(self.default_compression_ratio)],
            dtype=torch.float64,
        ).share_memory_()
        
        # Initialize FrameSkip components
        if self.enable_frameskip:
            self._init_frameskip()
    
    def _init_frameskip(self):
        """Initialize FrameSkip components."""
        self._frameskip_runtime_stats = {
            "vac_frame_extraction_failures": 0,
            "vac_trajectories_without_frames": 0,
        }

        # Create importance calculator
        importance_cfg = self.frameskip_config.get("importance", {})
        self.vac_video_backend = importance_cfg.get("video_backend", getattr(self, "video_backend", "decord"))
        self.vac_allow_backend_fallback = bool(importance_cfg.get("allow_backend_fallback", False))
        self.max_vac_frames = importance_cfg.get("max_vac_frames", None)
        self.max_vac_frames = int(self.max_vac_frames) if self.max_vac_frames else None
        importance_cfg_for_calculator = dict(importance_cfg)
        importance_cfg_for_calculator.pop("video_backend", None)
        importance_cfg_for_calculator.pop("allow_backend_fallback", None)
        importance_cfg_for_calculator.pop("max_vac_frames", None)
        self.importance_calculator = get_importance_calculator(importance_cfg_for_calculator)
        
        # Create frame pruner
        pruning_cfg = self.frameskip_config.get("pruning", {})
        self.frame_pruner = get_frame_pruner(pruning_cfg)
        
        # Create cache manager
        cache_dir = self.frameskip_config.get("cache_dir", None)
        self.trust_external_cache = cache_dir is not None
        self.cache_manager = FrameSkipCacheManager(
            dataset_path=self.dataset_path,
            config=self.frameskip_config,
            cache_dir=Path(cache_dir) / self.dataset_path.name if cache_dir else None,
        )
        
        # Load or compute frame importance
        self._load_or_compute_importance()
        
        # Update all_steps based on compressed data
        self._update_steps_with_compression()
    
    def _load_or_compute_importance(self):
        """Load cached importance or compute for all trajectories.

        In a distributed setting only rank 0 computes the cache; all other
        ranks poll the filesystem until the cache becomes valid.
        """
        import torch.distributed as dist

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        if self.trust_external_cache and self.cache_manager.has_any_cache():
            if rank == 0:
                try:
                    print(
                        f"[Rank {rank}] Reusing external FrameSkip cache without hash validation: "
                        f"{self.cache_manager.cache_dir}"
                    )
                    self.frameskip_cache = self.cache_manager.load_all_caches(skip_validation=True)
                    self._validate_requested_compression_ratios()
                    return
                except ValueError as err:
                    print(
                        f"[Rank {rank}] External FrameSkip cache is incomplete for requested ratios; "
                        f"rebuilding cache at {self.cache_manager.cache_dir}. Error: {err}"
                    )
                    self.cache_manager.clear_cache()
            else:
                self._wait_for_cache_from_rank0(rank)
                print(f"[Rank {rank}] Loading FrameSkip cache from {self.cache_manager.cache_dir}")
                self.frameskip_cache = self.cache_manager.load_all_caches(skip_validation=True)
                self._validate_requested_compression_ratios()
                return

        if self.cache_manager.is_cache_valid():
            if rank == 0:
                try:
                    print(f"[Rank {rank}] Loading FrameSkip cache from {self.cache_manager.cache_dir}")
                    self.frameskip_cache = self.cache_manager.load_all_caches()
                    self._validate_requested_compression_ratios()
                    return
                except ValueError as err:
                    print(
                        f"[Rank {rank}] Existing FrameSkip cache is incompatible with requested ratios; "
                        f"rebuilding cache at {self.cache_manager.cache_dir}. Error: {err}"
                    )
                    self.cache_manager.clear_cache()
            else:
                self._wait_for_cache_from_rank0(rank)
                print(f"[Rank {rank}] Loading FrameSkip cache from {self.cache_manager.cache_dir}")
                self.frameskip_cache = self.cache_manager.load_all_caches(
                    skip_validation=self.trust_external_cache
                )
                self._validate_requested_compression_ratios()
                return

        # Cache does not exist — only rank 0 computes to avoid OOM from
        # all processes loading frames + running the visual encoder in parallel.
        if rank == 0:
            print("Computing frame importance for all trajectories (rank 0 only)...")
            self.frameskip_cache, num_failed, importance_stats = self._compute_all_importance()
            import platform, torch
            extra_metadata = {
                "dataset_path": str(self.dataset_path),
                "video_backend": getattr(self, "video_backend", "decord"),
                "device": self.frameskip_config.get("importance", {}).get("device", "cuda"),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "importance_stats": importance_stats,
            }
            self.cache_manager.save_all_caches(
                self.frameskip_cache,
                num_failed=num_failed,
                extra_metadata=extra_metadata,
            )
            print(f"FrameSkip cache saved to {self.cache_manager.cache_dir}")
            self._validate_requested_compression_ratios()

        if rank != 0:
            self._wait_for_cache_from_rank0(rank)
            print(f"[Rank {rank}] Loading FrameSkip cache from {self.cache_manager.cache_dir}")
            self.frameskip_cache = self.cache_manager.load_all_caches()
            self._validate_requested_compression_ratios()

    def _normalize_ratio(self, ratio) -> float:
        """Normalize ratio values for stable comparison and messaging."""
        return round(float(ratio), 6)

    def _get_active_training_ratios(self) -> List[float]:
        """Return the compression ratios that this training run may actually use."""
        pruning_cfg = self.frameskip_config.get("pruning", {})
        requested_ratios = pruning_cfg.get("used_compression_ratios", None)
        if requested_ratios in (None, []):
            requested_ratios = pruning_cfg.get("compression_ratios", [self.default_compression_ratio])
        return [self._normalize_ratio(ratio) for ratio in requested_ratios]

    def _get_cached_ratios(self) -> List[float]:
        """Return the compression ratios available in the loaded cache."""
        metadata = self.cache_manager.load_metadata()
        pruning_cfg = metadata.get("pruning", {})
        metadata_ratios = pruning_cfg.get("compression_ratios", None)
        if metadata_ratios:
            return [self._normalize_ratio(ratio) for ratio in metadata_ratios]

        ratio_set = set()
        for traj_cache in self.frameskip_cache.values():
            if isinstance(traj_cache, dict):
                ratio_set.update(self._normalize_ratio(ratio) for ratio in traj_cache.keys())
            if ratio_set:
                break

        return sorted(ratio_set)

    def _validate_requested_compression_ratios(self):
        """Fail early if training asks for compression ratios missing from the cache."""
        requested_ratios = self._get_active_training_ratios()
        cached_ratios = self._get_cached_ratios()

        if not cached_ratios:
            return

        cached_ratio_set = set(cached_ratios)
        missing_ratios = [ratio for ratio in requested_ratios if ratio not in cached_ratio_set]
        if not missing_ratios:
            return

        raise ValueError(
            "FrameSkip cache does not contain all requested training compression ratios. "
            f"Requested: {requested_ratios}. "
            f"Missing from cache: {missing_ratios}. "
            f"Available in cache: {cached_ratios}. "
            "Either regenerate cache with a larger pruning.compression_ratios superset, "
            "or restrict pruning.used_compression_ratios to ratios already present in the cache."
        )

    def _wait_for_cache_from_rank0(self, rank: int):
        """Wait for rank 0 to finish cache generation without using a long NCCL barrier."""
        poll_interval_seconds = float(self.frameskip_config.get("cache_poll_interval_seconds", 10.0))
        max_wait_config = self.frameskip_config.get("cache_wait_timeout_seconds", 7 * 24 * 3600)
        max_wait_seconds = float(max_wait_config) if max_wait_config is not None else 0.0
        start_time = time.time()
        last_log_time = 0.0

        while True:
            if self.trust_external_cache:
                if self.cache_manager.has_any_cache():
                    metadata = self.cache_manager.load_metadata()
                    cached_ratios = metadata.get("pruning", {}).get("compression_ratios", [])
                    requested_ratios = self._get_active_training_ratios()
                    if cached_ratios and all(
                        self._normalize_ratio(ratio) in {self._normalize_ratio(r) for r in cached_ratios}
                        for ratio in requested_ratios
                    ):
                        return
                else:
                    pass
            elif self.cache_manager.is_cache_valid():
                return

            elapsed = time.time() - start_time
            if max_wait_seconds > 0 and elapsed > max_wait_seconds:
                raise TimeoutError(
                    f"[Rank {rank}] Timed out waiting for FrameSkip cache after {max_wait_seconds:.0f}s: "
                    f"{self.cache_manager.cache_dir}"
                )

            if elapsed - last_log_time >= 60.0:
                timeout_desc = f"{max_wait_seconds / 3600.0:.1f} h timeout" if max_wait_seconds > 0 else "no timeout"
                print(
                    f"[Rank {rank}] Waiting for rank 0 to finish FrameSkip cache generation "
                    f"({elapsed / 60.0:.1f} min elapsed, {timeout_desc}): {self.cache_manager.cache_dir}"
                )
                last_log_time = elapsed

            time.sleep(poll_interval_seconds)
    
    def _compute_all_importance(self) -> tuple:
        """
        Compute frame importance for all trajectories.

        Returns:
            Tuple of (all_caches, num_failed, importance_stats) where all_caches maps
            trajectory_id to compressed data at different ratios, num_failed is the
            count of trajectories that fell back to identity mapping due to errors,
            and importance_stats records VAC-related extraction failures.
        """
        from tqdm import tqdm

        all_caches = {}
        num_failed = 0

        for trajectory_id in tqdm(self.trajectory_ids, desc="Computing Frame Importance"):
            try:
                # Load full trajectory
                trajectory = self._load_trajectory_for_importance(trajectory_id)

                # Compute importance scores
                importance_scores = self.importance_calculator.compute_importance(trajectory)

                # Create compressed versions at different ratios
                compressed_data = self.frame_pruner.create_multi_ratio_trajectory(
                    trajectory, importance_scores
                )

                # Store in cache (convert to simple dict for storage)
                all_caches[str(trajectory_id)] = {
                    ratio: {
                        "keep_indices": result.keep_indices,
                        "importance_scores": result.importance_scores,
                        "compression_ratio": result.compression_ratio,
                        "actual_ratio": result.actual_ratio,
                    }
                    for ratio, result in compressed_data.items()
                }

            except Exception as e:
                num_failed += 1
                # Get trajectory length safely
                try:
                    traj_idx = np.where(self.trajectory_ids == trajectory_id)[0]
                    if len(traj_idx) > 0:
                        traj_length = self.trajectory_lengths[traj_idx[0]]
                    else:
                        # Fallback: try to get from loaded trajectory
                        traj_data = self.get_trajectory_data(trajectory_id)
                        traj_length = len(traj_data) if traj_data is not None else 10
                except Exception:
                    traj_length = 10  # Default fallback

                all_caches[str(trajectory_id)] = {
                    1.0: {
                        "keep_indices": np.arange(traj_length),
                        "importance_scores": np.ones(traj_length),
                        "compression_ratio": 1.0,
                        "actual_ratio": 1.0,
                    }
                }

        importance_stats = {
            "vac_enabled": bool(getattr(self.importance_calculator, "use_visual_encoder", False)),
            "vac_frame_extraction_failures": int(
                self._frameskip_runtime_stats.get("vac_frame_extraction_failures", 0)
            ),
            "vac_trajectories_without_frames": int(
                self._frameskip_runtime_stats.get("vac_trajectories_without_frames", 0)
            ),
            "vac_trajectory_failure_rate": (
                float(self._frameskip_runtime_stats.get("vac_trajectories_without_frames", 0))
                / float(len(self.trajectory_ids))
                if len(self.trajectory_ids) > 0 and getattr(self.importance_calculator, "use_visual_encoder", False)
                else 0.0
            ),
        }

        return all_caches, num_failed, importance_stats
    
    def _load_trajectory_for_importance(self, trajectory_id: int) -> Dict[str, Any]:
        """
        Load trajectory data for importance computation.
        
        This loads raw trajectory data without transforms.
        """
        # Get trajectory data
        traj_data = self.get_trajectory_data(trajectory_id)
        
        # Extract actions
        actions = []
        for action_key in self.modality_keys.get("action", []):
            action_data = self._extract_action_from_traj(traj_data, action_key)
            if action_data is not None:
                actions.append(action_data)
        
        if actions:
            actions = np.concatenate(actions, axis=1)
        else:
            # Fallback: try to get from trajectory directly
            actions = self._get_actions_from_traj_data(traj_data)
        
        # Extract frames if needed for visual encoder
        observations = []
        if self.importance_calculator.use_visual_encoder:
            video_keys = self.modality_keys.get("video", [])
            for video_key in self.modality_keys.get("video", []):
                frames = self._extract_frames_from_traj(
                    trajectory_id=trajectory_id,
                    video_key=video_key,
                    traj_data=traj_data,
                    target_length=len(actions),
                )
                if frames:
                    observations = frames
                    break

            if video_keys and not observations:
                self._frameskip_runtime_stats["vac_trajectories_without_frames"] += 1
        
        return {
            "observations": observations,
            "actions": actions,
            "trajectory_id": trajectory_id,
        }
    
    def _extract_action_from_traj(self, traj_data, action_key: str) -> Optional[np.ndarray]:
        """Extract action data from trajectory using lerobot modality meta mapping."""
        try:
            # action_key is like "action.x" — strip prefix to get subkey
            subkey = action_key.replace("action.", "")
            le_action_cfg = self.lerobot_modality_meta.action
            if subkey not in le_action_cfg:
                return None
            
            action_meta = le_action_cfg[subkey]
            original_key = action_meta.original_key or subkey
            if original_key not in traj_data.columns:
                return None
            
            # Extract data
            data_values = traj_data[original_key].values
            if len(data_values) == 0:
                return None
            
            # Stack the data - handle both scalar and array values
            first_item = data_values[0]
            if np.isscalar(first_item):
                # Scalar values (e.g., gripper)
                data_array = np.array(data_values).reshape(-1, 1)
            else:
                # Array values (e.g., position)
                data_array = np.stack([np.array(x) for x in data_values])
                if data_array.ndim == 1:
                    data_array = data_array.reshape(-1, 1)
            
            # Get indices safely
            start_idx = getattr(action_meta, 'start', 0)
            end_idx = getattr(action_meta, 'end', data_array.shape[-1] if data_array.ndim > 1 else 1)
            
            if start_idx >= end_idx or start_idx >= data_array.shape[-1]:
                # Invalid indices, return full array
                return data_array
            
            # Ensure end_idx doesn't exceed array bounds
            end_idx = min(end_idx, data_array.shape[-1])
            
            le_indices = np.arange(start_idx, end_idx)
            if data_array.ndim > 1:
                return data_array[:, le_indices]
            else:
                return data_array[le_indices].reshape(-1, 1)
        except Exception as e:
            # Log error for debugging if needed
            return None
    
    def _get_actions_from_traj_data(self, traj_data) -> np.ndarray:
        """Fallback method to get actions from trajectory data."""
        if traj_data is None or len(traj_data) == 0:
            return np.zeros((1, 7))  # Return minimal dummy
        
        traj_len = len(traj_data)
        
        # Try common action column names
        action_candidates = [
            "action", "actions", "action.eef_position", "action.gripper_close",
            "action.x", "action.y", "action.z"
        ]
        
        actions_list = []
        for col in action_candidates:
            if hasattr(traj_data, 'columns') and col in traj_data.columns:
                try:
                    data = traj_data[col]
                    if hasattr(data, 'values'):
                        values = data.values
                        if len(values) == 0:
                            continue
                        
                        first_item = values[0]
                        if np.isscalar(first_item):
                            # Scalar values
                            action_array = np.array(values).reshape(-1, 1)
                        else:
                            # Array values
                            action_array = np.stack([np.array(x) for x in values])
                            if action_array.ndim == 1:
                                action_array = action_array.reshape(-1, 1)
                        
                        if len(action_array) == traj_len:
                            actions_list.append(action_array)
                except Exception:
                    continue
        
        if actions_list:
            try:
                return np.concatenate(actions_list, axis=1)
            except Exception:
                # If concatenation fails, return the first one
                return actions_list[0]
        
        # Try to infer action dim from stats
        action_dim = 7  # default
        try:
            if hasattr(self, 'stats') and self.stats and 'action' in self.stats:
                action_stats = self.stats['action']
                if 'mean' in action_stats:
                    action_dim = len(action_stats['mean'])
        except Exception:
            pass
        
        # Return dummy actions
        return np.zeros((traj_len, action_dim))
    
    def _extract_frames_from_traj(
        self,
        trajectory_id: int,
        video_key: str,
        traj_data=None,
        target_length: Optional[int] = None,
    ) -> List[Image.Image]:
        """Extract trajectory-aligned frames for visual importance computation."""
        video_path = self.get_video_path(trajectory_id, video_key.replace("video.", ""))
        base_video_backend_kwargs = getattr(self, "video_backend_kwargs", None) or {}

        timestamps = None
        if traj_data is not None and hasattr(traj_data, "columns") and "timestamp" in traj_data.columns:
            timestamps = traj_data["timestamp"].to_numpy()
            if target_length is not None and len(timestamps) > target_length:
                timestamps = timestamps[:target_length]
            if self.max_vac_frames is not None and len(timestamps) > self.max_vac_frames:
                sample_idx = np.linspace(0, len(timestamps) - 1, self.max_vac_frames, dtype=int)
                timestamps = timestamps[sample_idx]

        backend_candidates = []
        for backend_name in [self.vac_video_backend]:
            if backend_name and backend_name not in backend_candidates:
                backend_candidates.append(backend_name)
        if self.vac_allow_backend_fallback:
            for backend_name in [getattr(self, "video_backend", "decord"), "opencv", "pyav"]:
                if backend_name and backend_name not in backend_candidates:
                    backend_candidates.append(backend_name)

        last_error = None
        for backend_name in backend_candidates:
            try:
                video_backend_kwargs = base_video_backend_kwargs if backend_name == getattr(self, "video_backend", "decord") else {}

                if timestamps is not None and len(timestamps) > 0:
                    from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps

                    frames = get_frames_by_timestamps(
                        video_path.as_posix(),
                        timestamps,
                        video_backend=backend_name,
                        video_backend_kwargs=video_backend_kwargs,
                    )
                else:
                    from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_indices

                    traj_idx = np.where(self.trajectory_ids == trajectory_id)[0]
                    traj_length = int(self.trajectory_lengths[traj_idx[0]]) if len(traj_idx) > 0 else int(target_length or 1)
                    sample_length = int(target_length or traj_length)
                    if self.max_vac_frames is not None:
                        sample_length = min(sample_length, self.max_vac_frames)
                    sample_length = max(1, min(sample_length, traj_length))
                    sample_indices = np.linspace(0, max(traj_length - 1, 0), sample_length, dtype=int)
                    frames = get_frames_by_indices(
                        video_path.as_posix(),
                        sample_indices,
                        video_backend=backend_name,
                        video_backend_kwargs=video_backend_kwargs,
                    )

                pil_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        pil_frames.append(Image.fromarray(frame))
                    else:
                        pil_frames.append(frame)

                if backend_name != self.vac_video_backend:
                    print(
                        f"[FrameSkip] VAC fallback backend succeeded for traj={trajectory_id}: "
                        f"{self.vac_video_backend} -> {backend_name}"
                    )
                return pil_frames
            except Exception as e:
                last_error = e
                self._frameskip_runtime_stats["vac_frame_extraction_failures"] += 1
                continue

        print(f"Warning: Failed to extract frames for {trajectory_id}: {last_error}")
        return []
    
    def _update_steps_with_compression(self):
        """
        Update all_steps to reflect compressed frame counts.
        
        For each trajectory and compression ratio, create step entries.
        """
        if not self.enable_frameskip:
            return
        
        # For now, we'll keep the original steps but use compression at access time
        # This maintains compatibility with the base dataset
        # The actual compression happens in __getitem__
    
    def get_compressed_step_indices(
        self,
        trajectory_id: int,
        base_index: int,
        compression_ratio: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Get compressed step indices for a given base index.
        
        Args:
            trajectory_id: Trajectory ID
            base_index: Base step index (in original trajectory)
            compression_ratio: Compression ratio to use
            
        Returns:
            Tuple of (compressed_index, keep_indices)
        """
        ratio = compression_ratio or self.default_compression_ratio
        
        # Get cached data for this trajectory
        traj_cache = self.frameskip_cache.get(str(trajectory_id), {})
        
        # Find closest available ratio
        available_ratios = sorted(traj_cache.keys())
        if not available_ratios:
            # No cache available, return full trajectory
            try:
                traj_idx = np.where(self.trajectory_ids == trajectory_id)[0]
                if len(traj_idx) > 0:
                    traj_length = self.trajectory_lengths[traj_idx[0]]
                else:
                    # Fallback: use a reasonable default
                    traj_length = base_index + 1
            except Exception:
                traj_length = base_index + 1
            return base_index, np.arange(traj_length)
        
        closest_ratio = min(available_ratios, key=lambda x: abs(x - ratio))
        ratio_data = traj_cache[closest_ratio]
        
        keep_indices = ratio_data["keep_indices"]
        
        # Map base_index to compressed index
        # Find the position in keep_indices that is closest to base_index
        compressed_idx = np.searchsorted(keep_indices, base_index)
        compressed_idx = np.clip(compressed_idx, 0, len(keep_indices) - 1)
        
        return compressed_idx, keep_indices
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a training sample with FrameSkip applied.

        Args:
            index: Sample index

        Returns:
            Training sample dictionary
        """
        if not self.enable_frameskip:
            return super().__getitem__(index)

        # Get trajectory and base index from original steps
        trajectory_id, base_index = self.all_steps[index]

        # Get current compression ratio. This value is stored in shared memory
        # so DataLoader worker processes see updates made by the trainer.
        compression_ratio = self.get_compression_ratio()

        # Map base_index to the nearest kept frame index
        _, keep_indices = self.get_compressed_step_indices(
            trajectory_id, base_index, compression_ratio
        )
        compressed_idx = np.searchsorted(keep_indices, base_index)
        compressed_idx = int(np.clip(compressed_idx, 0, len(keep_indices) - 1))
        actual_index = int(keep_indices[compressed_idx])

        # Directly fetch data at the mapped index — no all_steps mutation
        if base_index != actual_index and np.random.random() < 0.001:
            print(f"[FrameSkip] traj={trajectory_id}, base={base_index} -> actual={actual_index}, ratio={compression_ratio:.1f}, skipped={base_index - actual_index:+d}")

        raw_data = self.get_step_data(trajectory_id, actual_index)
        data = self.transforms(raw_data)
        data["compression_ratio"] = float(compression_ratio)
        data["frameskip_base_index"] = int(base_index)
        data["frameskip_actual_index"] = int(actual_index)

        return data

    def get_compression_ratio(self) -> float:
        """Return the active compression ratio visible to DataLoader workers."""
        if hasattr(self, "_compression_ratio_tensor"):
            return float(self._compression_ratio_tensor[0].item())
        return float(getattr(self, "current_compression_ratio", self.default_compression_ratio))
    
    def set_compression_ratio(self, ratio: float):
        """
        Set the current compression ratio.
        
        Args:
            ratio: Compression ratio between 0 and 1
        """
        clipped_ratio = float(np.clip(ratio, 0.1, 1.0))
        self.current_compression_ratio = clipped_ratio
        if hasattr(self, "_compression_ratio_tensor"):
            self._compression_ratio_tensor[0] = clipped_ratio


class FrameSkipMixtureDataset(LeRobotMixtureDataset):
    """
    LeRobotMixtureDataset with FrameSkip support.
    
    Extends the mixture dataset to support frame skipping.
    """
    
    def __init__(
        self,
        data_mixture: Sequence[Tuple[LeRobotSingleDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        frameskip_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize FrameSkip mixture dataset.
        
        Args:
            data_mixture: List of (dataset, weight) tuples
            mode: "train", "val", or "test"
            balance_dataset_weights: Whether to balance dataset weights
            balance_trajectory_weights: Whether to balance trajectory weights
            seed: Random seed
            frameskip_config: FrameSkip configuration
            **kwargs: Additional arguments
        """
        # Wrap datasets with FrameSkip if needed
        wrapped_mixture = []
        for dataset, weight in data_mixture:
            if not isinstance(dataset, FrameSkipSingleDataset):
                # Create FrameSkip version
                frameskip_dataset = self._create_frameskip_dataset(
                    dataset, frameskip_config
                )
                wrapped_mixture.append((frameskip_dataset, weight))
            else:
                wrapped_mixture.append((dataset, weight))
        
        # Initialize parent
        super().__init__(
            data_mixture=wrapped_mixture,
            mode=mode,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
            **kwargs,
        )
        
        self.frameskip_config = frameskip_config
        self.enable_frameskip = frameskip_config.get("enabled", False) if frameskip_config else False
    
    def _create_frameskip_dataset(
        self,
        dataset: LeRobotSingleDataset,
        frameskip_config: Optional[Dict],
    ) -> FrameSkipSingleDataset:
        """
        Create a FrameSkip version of a dataset.
        
        This creates a new instance with the same configuration plus FrameSkip.
        """
        # Create new instance with FrameSkip enabled
        frameskip_dataset = FrameSkipSingleDataset(
            dataset_path=dataset.dataset_path,
            modality_configs=dataset.modality_configs,
            embodiment_tag=dataset.tag,
            video_backend=dataset.video_backend,
            video_backend_kwargs=getattr(dataset, "video_backend_kwargs", None),
            transforms=dataset.transforms,
            delete_pause_frame=getattr(dataset, "delete_pause_frame", False),
            data_cfg=getattr(dataset, "data_cfg", None),
            frameskip_config=frameskip_config,
        )
        
        return frameskip_dataset
    
    def set_compression_ratio(self, ratio: float):
        """Set compression ratio for all datasets."""
        for dataset in self.datasets:
            if isinstance(dataset, FrameSkipSingleDataset):
                dataset.set_compression_ratio(ratio)
    
    def get_frameskip_stats(self) -> Dict[str, Any]:
        """Get statistics about FrameSkip."""
        if not self.enable_frameskip:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "num_datasets": len(self.datasets),
            "datasets_with_frameskip": sum(
                1 for d in self.datasets if isinstance(d, FrameSkipSingleDataset)
            ),
        }
        
        # Per-dataset stats
        dataset_stats = []
        for dataset in self.datasets:
            if isinstance(dataset, FrameSkipSingleDataset):
                cache_stats = dataset.cache_manager.get_cache_stats()
                dataset_stats.append({
                    "name": dataset.dataset_name,
                    "cache": cache_stats,
                })
        
        stats["datasets"] = dataset_stats
        return stats


def create_frameskip_dataset(
    base_dataset: LeRobotSingleDataset,
    frameskip_config: Dict[str, Any],
) -> FrameSkipSingleDataset:
    """
    Factory function to create FrameSkip dataset from base dataset.
    
    Args:
        base_dataset: Base LeRobotSingleDataset
        frameskip_config: FrameSkip configuration
        
    Returns:
        FrameSkipSingleDataset instance
    """
    return FrameSkipSingleDataset(
        dataset_path=base_dataset.dataset_path,
        modality_configs=base_dataset.modality_configs,
        embodiment_tag=base_dataset.tag,
        video_backend=base_dataset.video_backend,
        video_backend_kwargs=getattr(base_dataset, "video_backend_kwargs", None),
        transforms=base_dataset.transforms,
        delete_pause_frame=getattr(base_dataset, "delete_pause_frame", False),
        data_cfg=getattr(base_dataset, "data_cfg", None),
        frameskip_config=frameskip_config,
    )
