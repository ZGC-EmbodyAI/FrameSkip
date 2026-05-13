# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Cache Manager for FrameSkip.

Manages caching of frame importance scores and compressed trajectories
to avoid recomputation.
"""

import os
import pickle
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from omegaconf import OmegaConf


class FrameSkipCacheManager:
    """
    Manages caching of frame importance scores and pruning results.
    
    Cache structure:
    dataset_root/
    └── .frameskip_cache/
        ├── metadata.json          # Cache metadata (version, creation time)
        ├── config_hash.pkl        # Cached configuration
        └── trajectories/
            ├── traj_0000.pkl      # Per-trajectory cache
            ├── traj_0001.pkl
            └── ...
    """
    
    CACHE_VERSION = "1.0.0"
    CACHE_DIR_NAME = ".frameskip_cache"
    
    def __init__(
        self,
        dataset_path: Path,
        config: Dict[str, Any],
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize cache manager.
        
        Args:
            dataset_path: Path to the dataset
            config: Configuration dictionary for hashing
            cache_dir: Optional custom cache directory
        """
        self.dataset_path = Path(dataset_path)
        self.config = self._to_plain_python(config)
        self.hash_config = self._build_hash_config(self.config)
        
        if cache_dir is None:
            self.cache_dir = self.dataset_path / self.CACHE_DIR_NAME
        else:
            self.cache_dir = Path(cache_dir)
        
        self.trajectories_dir = self.cache_dir / "trajectories"
        
        # Compute config hash for cache validation
        self.config_hash = self._compute_config_hash(self.hash_config)
        
        # Ensure cache directory exists
        self._ensure_cache_structure()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration for cache validation."""
        # Convert config to stable string representation
        config_str = json.dumps(self._to_plain_python(config), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _build_hash_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build the subset of config that should invalidate cache entries."""
        hash_config = json.loads(json.dumps(config, sort_keys=True, default=str))
        pruning_cfg = hash_config.get("pruning", {})
        pruning_cfg.pop("used_compression_ratios", None)
        return hash_config

    def _to_plain_python(self, value: Any) -> Any:
        """Convert tracked config objects and numpy values into JSON-safe Python objects."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if hasattr(value, "unwrap"):
            try:
                value = value.unwrap()
            except Exception:
                pass

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)

        if isinstance(value, dict):
            return {str(k): self._to_plain_python(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._to_plain_python(v) for v in value]

        if hasattr(value, "items"):
            try:
                return {str(k): self._to_plain_python(v) for k, v in value.items()}
            except Exception:
                pass

        return value
    
    def _ensure_cache_structure(self):
        """Create cache directory structure if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file if not exists
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            metadata = {
                "frameskip_version": self.CACHE_VERSION,
                "created_at": datetime.now().isoformat(),
                "config_hash": self.config_hash,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
    
    def is_cache_valid(self) -> bool:
        """Check if existing cache is valid (matching config and non-empty)."""
        metadata_path = self.cache_dir / "metadata.json"

        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            if metadata.get("frameskip_version") != self.CACHE_VERSION:
                return False

            if metadata.get("config_hash") != self.config_hash:
                return False

            # Must have at least one trajectory cached
            cached_files = list(self.trajectories_dir.glob("*.pkl"))
            if len(cached_files) == 0:
                return False

            return True
        except Exception:
            return False

    def has_any_cache(self) -> bool:
        """Check whether the cache directory already contains trajectory cache files."""
        if not self.trajectories_dir.exists():
            return False
        return any(self.trajectories_dir.glob("*.pkl"))

    def load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata if available."""
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def get_trajectory_cache_path(self, trajectory_id: str) -> Path:
        """Get cache file path for a trajectory."""
        return self.trajectories_dir / f"{trajectory_id}.pkl"
    
    def has_trajectory_cache(self, trajectory_id: str) -> bool:
        """Check if a trajectory is cached."""
        cache_path = self.get_trajectory_cache_path(trajectory_id)
        return cache_path.exists()
    
    def load_trajectory_cache(
        self,
        trajectory_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached data for a trajectory.
        
        Args:
            trajectory_id: Trajectory identifier
            
        Returns:
            Cached data or None if not found
        """
        cache_path = self.get_trajectory_cache_path(trajectory_id)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache for {trajectory_id}: {e}")
            return None
    
    def save_trajectory_cache(
        self,
        trajectory_id: str,
        data: Dict[str, Any],
    ):
        """
        Save data to cache for a trajectory.
        
        Args:
            trajectory_id: Trajectory identifier
            data: Data to cache
        """
        cache_path = self.get_trajectory_cache_path(trajectory_id)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache for {trajectory_id}: {e}")
    
    def load_all_caches(self, skip_validation: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Load all cached trajectory data.
        
        Returns:
            Dictionary mapping trajectory_id to cached data
        """
        all_caches = {}
        
        if not skip_validation and not self.is_cache_valid():
            return all_caches
        
        for cache_file in self.trajectories_dir.glob("*.pkl"):
            trajectory_id = cache_file.stem
            data = self.load_trajectory_cache(trajectory_id)
            if data is not None:
                all_caches[trajectory_id] = data
        
        return all_caches
    
    def save_all_caches(
        self,
        all_data: Dict[str, Dict[str, Any]],
        num_failed: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save all trajectory caches.

        Args:
            all_data: Dictionary mapping trajectory_id to data
            num_failed: Number of trajectories that fell back to identity mapping
            extra_metadata: Additional fields to merge into metadata.json
                            (e.g. importance / pruning sub-configs, device info)
        """
        for trajectory_id, data in all_data.items():
            self.save_trajectory_cache(trajectory_id, data)

        # Calculate total cache size
        total_size = sum(
            f.stat().st_size for f in self.trajectories_dir.glob("*.pkl")
        )

        self._update_metadata(
            num_cached=len(all_data),
            num_failed=num_failed,
            total_size_mb=total_size / (1024 * 1024),
            extra_metadata=extra_metadata,
        )
    
    def _update_metadata(
        self,
        num_cached: int = 0,
        num_failed: int = 0,
        total_size_mb: float = 0.0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Write (or overwrite) cache metadata with full config details."""
        importance_cfg = self.config.get("importance", {})
        pruning_cfg = self.config.get("pruning", {})
        hash_pruning_cfg = self.hash_config.get("pruning", {})

        metadata: Dict[str, Any] = {
            "frameskip_version": self.CACHE_VERSION,
            "config_hash": self.config_hash,
            "created_at": datetime.now().isoformat(),
            "num_trajectories_cached": num_cached,
            "num_failed": num_failed,
            "total_size_mb": round(total_size_mb, 2),
            # ---- importance sub-config ----
            "importance": {
                "type": importance_cfg.get("type", "default"),
                "alpha": importance_cfg.get("alpha", 0.6),
                "beta": importance_cfg.get("beta", 0.2),
                "gamma": importance_cfg.get("gamma", 0.2),
                "use_visual_encoder": importance_cfg.get("use_visual_encoder", False),
                "visual_encoder_name": importance_cfg.get("visual_encoder_name", None),
                "visual_encoder_checkpoint": importance_cfg.get("visual_encoder_checkpoint", None),
                "visual_encoder_checkpoint_present": (
                    Path(importance_cfg["visual_encoder_checkpoint"]).exists()
                    if importance_cfg.get("visual_encoder_checkpoint")
                    else False
                ),
                "device": importance_cfg.get("device", "cuda"),
            },
            # ---- pruning sub-config ----
            "pruning": {
                "type": pruning_cfg.get("type", "default"),
                "compression_ratios": pruning_cfg.get("compression_ratios", [1.0]),
                "used_compression_ratios": pruning_cfg.get("used_compression_ratios", None),
                "preserve_key_stages": pruning_cfg.get("preserve_key_stages", True),
                "min_frames_per_trajectory": pruning_cfg.get("min_frames_per_trajectory", 5),
                "key_stage_boost": pruning_cfg.get("key_stage_boost", 1.5),
                "max_gap": pruning_cfg.get("max_gap", 5),
            },
            "hash_pruning": {
                "compression_ratios": hash_pruning_cfg.get("compression_ratios", [1.0]),
            },
            "cache_dir": str(self.cache_dir),
        }

        if extra_metadata:
            metadata.update(self._to_plain_python(extra_metadata))

        metadata_path = self.cache_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
        self._ensure_cache_structure()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"exists": False}
        
        num_cached = len(list(self.trajectories_dir.glob("*.pkl")))
        
        # Calculate total size
        total_size = 0
        for cache_file in self.trajectories_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
        
        return {
            "exists": True,
            "valid": self.is_cache_valid(),
            "num_trajectories_cached": num_cached,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


class SharedCacheManager:
    """
    Shared cache manager for multiple datasets.
    
    Useful when multiple datasets share the same importance calculation
    configuration.
    """
    
    def __init__(self, root_cache_dir: Path):
        """
        Initialize shared cache manager.
        
        Args:
            root_cache_dir: Root directory for all caches
        """
        self.root_cache_dir = Path(root_cache_dir)
        self.root_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._managers: Dict[str, FrameSkipCacheManager] = {}
    
    def get_manager(
        self,
        dataset_path: Path,
        config: Dict[str, Any],
    ) -> FrameSkipCacheManager:
        """
        Get or create cache manager for a dataset.
        
        Args:
            dataset_path: Path to dataset
            config: Configuration
            
        Returns:
            FrameSkipCacheManager instance
        """
        # Use dataset path as key
        dataset_key = str(dataset_path.resolve())
        
        if dataset_key not in self._managers:
            # Create cache subdirectory for this dataset
            dataset_name = Path(dataset_path).name
            cache_dir = self.root_cache_dir / dataset_name
            
            self._managers[dataset_key] = FrameSkipCacheManager(
                dataset_path=dataset_path,
                config=config,
                cache_dir=cache_dir,
            )
        
        return self._managers[dataset_key]
    
    def clear_all_caches(self):
        """Clear all cached data."""
        import shutil
        if self.root_cache_dir.exists():
            shutil.rmtree(self.root_cache_dir)
        self.root_cache_dir.mkdir(parents=True, exist_ok=True)
        self._managers.clear()


def create_cache_manager(
    dataset_path: Path,
    config: Dict[str, Any],
    shared_root: Optional[Path] = None,
) -> FrameSkipCacheManager:
    """
    Factory function to create appropriate cache manager.
    
    Args:
        dataset_path: Path to dataset
        config: Configuration
        shared_root: Optional shared cache root
        
    Returns:
        Cache manager instance
    """
    if shared_root is not None:
        shared_manager = SharedCacheManager(shared_root)
        return shared_manager.get_manager(dataset_path, config)
    else:
        return FrameSkipCacheManager(dataset_path, config)
