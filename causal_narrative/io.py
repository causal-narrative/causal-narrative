"""
I/O utilities: caching, checkpoints, and file format helpers.

This module combines:
- Caching utilities for expensive operations
- Checkpoint and resume functionality for pipeline execution
- Parquet and JSONL file format helpers
"""

import hashlib
import json
import pickle
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import pandas as pd
from loguru import logger
from pydantic import BaseModel


T = TypeVar("T")


# =============================================================================
# Caching
# =============================================================================

def _compute_hash(obj: Any) -> str:
    """Compute hash of an object for cache key."""
    try:
        serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(serialized).hexdigest()
    except (TypeError, AttributeError) as e:
        logger.warning(f"Could not pickle object for hashing, using str(): {e}")
        return hashlib.sha256(str(obj).encode()).hexdigest()


def cache_result(
    cache_dir: Optional[Path] = None,
    key_prefix: str = "",
    use_args: bool = True,
    use_kwargs: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to cache function results to disk.

    Args:
        cache_dir: Directory to store cache files (default: .cache/)
        key_prefix: Prefix for cache keys
        use_args: Include positional arguments in cache key
        use_kwargs: Include keyword arguments in cache key

    Returns:
        Decorated function with caching
    """
    if cache_dir is None:
        cache_dir = Path(".cache")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_dir.mkdir(parents=True, exist_ok=True)

            key_parts = [key_prefix, func.__name__]

            if use_args and args:
                args_hash = _compute_hash(args)
                key_parts.append(f"args_{args_hash[:16]}")

            if use_kwargs and kwargs:
                kwargs_hash = _compute_hash(kwargs)
                key_parts.append(f"kwargs_{kwargs_hash[:16]}")

            cache_key = "_".join(key_parts)
            cache_file = cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    logger.info(f"Loaded cached result from {cache_file}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cache from {cache_file}: {e}")

            logger.info(f"Computing result for {func.__name__} (no cache found)")
            result = func(*args, **kwargs)

            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Cached result to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache to {cache_file}: {e}")

            return result

        return wrapper

    return decorator


def clear_cache(cache_dir: Optional[Path] = None, pattern: str = "*.pkl") -> int:
    """Clear cached files.

    Args:
        cache_dir: Directory containing cache files (default: .cache/)
        pattern: Glob pattern for files to delete

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = Path(".cache")

    if not cache_dir.exists():
        logger.info(f"Cache directory {cache_dir} does not exist")
        return 0

    deleted = 0
    for cache_file in cache_dir.glob(pattern):
        try:
            cache_file.unlink()
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file}: {e}")

    logger.info(f"Deleted {deleted} cache files from {cache_dir}")
    return deleted


def get_cache_size(cache_dir: Optional[Path] = None) -> int:
    """Get total size of cached files in bytes."""
    if cache_dir is None:
        cache_dir = Path(".cache")

    if not cache_dir.exists():
        return 0

    total_size = 0
    for cache_file in cache_dir.rglob("*"):
        if cache_file.is_file():
            total_size += cache_file.stat().st_size

    return total_size


# =============================================================================
# Checkpoint Management
# =============================================================================

class RunManifest(BaseModel):
    """Manifest tracking pipeline run state."""

    run_id: str
    config_hash: str
    file_hash: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    steps_completed: List[str] = []
    metadata: Dict[str, Any] = {}

    class Config:
        frozen = False


class CheckpointManager:
    """Manages checkpoints and resumption for pipeline runs.

    Attributes:
        run_dir: Directory for this run's outputs
        manifest_file: Path to run manifest file
        manifest: Current run manifest
    """

    def __init__(self, run_dir: Path, config: Optional[Any] = None):
        """Initialize checkpoint manager."""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.run_dir / "manifest.json"

        if self.manifest_file.exists():
            self.manifest = self._load_manifest()
            logger.info(f"Loaded existing manifest for run {self.manifest.run_id}")
        else:
            config_hash = self._compute_config_hash(config) if config else "no-config"
            self.manifest = RunManifest(
                run_id=self.run_dir.name,
                config_hash=config_hash,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self._save_manifest()
            logger.info(f"Created new manifest for run {self.manifest.run_id}")

    def _compute_config_hash(self, config: Any) -> str:
        """Compute hash of configuration."""
        try:
            if hasattr(config, "model_dump"):
                config_dict = config.model_dump()
            elif hasattr(config, "dict"):
                config_dict = config.dict()
            else:
                config_dict = vars(config)

            config_str = json.dumps(config_dict, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to compute config hash: {e}")
            return "unknown"

    def _load_manifest(self) -> RunManifest:
        """Load manifest from file."""
        with open(self.manifest_file, "r") as f:
            data = json.load(f)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            return RunManifest(**data)

    def _save_manifest(self) -> None:
        """Save manifest to file."""
        self.manifest.updated_at = datetime.now()
        with open(self.manifest_file, "w") as f:
            json.dump(
                self.manifest.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    def mark_step_complete(self, step_name: str) -> None:
        """Mark a pipeline step as completed."""
        if step_name not in self.manifest.steps_completed:
            self.manifest.steps_completed.append(step_name)
            self._save_manifest()
            logger.info(f"Marked step '{step_name}' as complete")

    def is_step_complete(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        return step_name in self.manifest.steps_completed

    def get_checkpoint_path(self, step_name: str, suffix: str = "parquet") -> Path:
        """Get path for a checkpoint file."""
        return self.run_dir / f"{step_name}.{suffix}"

    def save_checkpoint(
        self,
        step_name: str,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        mark_complete: bool = True,
    ) -> Path:
        """Save checkpoint data."""
        checkpoint_path = self.get_checkpoint_path(step_name)

        if isinstance(data, pd.DataFrame):
            save_parquet(data, checkpoint_path)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            save_parquet(df, checkpoint_path)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if mark_complete:
            self.mark_step_complete(step_name)

        logger.info(f"Saved checkpoint for step '{step_name}' to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, step_name: str) -> Optional[pd.DataFrame]:
        """Load checkpoint data."""
        checkpoint_path = self.get_checkpoint_path(step_name)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        df = load_parquet(checkpoint_path)
        logger.info(f"Loaded checkpoint for step '{step_name}' from {checkpoint_path}")
        return df

    def should_run_step(self, step_name: str, force_rerun: bool = False) -> bool:
        """Determine if a step should be run."""
        if force_rerun:
            return True

        return not self.is_step_complete(step_name)

    def update_metadata(self, key: str, value: Any) -> None:
        """Update run metadata."""
        self.manifest.metadata[key] = value
        self._save_manifest()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get run metadata."""
        return self.manifest.metadata.get(key, default)


# =============================================================================
# File Format Helpers
# =============================================================================

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.debug(f"Saved {len(df)} rows to {path}")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    logger.debug(f"Loaded {len(df)} rows from {path}")
    return df


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save data to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")
    logger.debug(f"Saved {len(data)} items to {path}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.debug(f"Loaded {len(data)} items from {path}")
    return data
