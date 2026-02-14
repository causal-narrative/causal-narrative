"""Configuration management for causal narrative lab."""

import os
from typing import List, Optional, Any, Dict
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class LLMConfig(BaseSettings):
    """LLM API configuration."""

    model_config = SettingsConfigDict(env_prefix="SILICONFLOW_", case_sensitive=False)

    base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="Base URL for SiliconFlow API"
    )
    api_keys: List[str] = Field(default_factory=list, description="List of API keys")
    model_name: str = Field(
        default="deepseek-ai/DeepSeek-V3",
        description="Model name to use"
    )
    max_workers: int = Field(default=4, description="Maximum concurrent workers")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=0.0, description="Sampling temperature")


class PipelineConfig(BaseSettings):
    """Pipeline execution configuration."""

    model_config = SettingsConfigDict(env_prefix="PIPELINE_")

    enable_cache: bool = Field(default=True, description="Enable result caching")
    force_rerun: bool = Field(default=False, description="Force rerun all steps")
    checkpoint_interval: int = Field(default=100, description="Checkpoint frequency")


class SRLConfig(BaseSettings):
    """Semantic Role Labeling configuration."""

    model_config = SettingsConfigDict(env_prefix="SRL_")

    backend: str = Field(default="spacy", description="SRL backend: 'spacy' or 'allennlp'")
    spacy_model: str = Field(default="en_core_web_sm", description="spaCy model name")
    batch_size: int = Field(default=32, description="Batch processing size")


class ClusteringConfig(BaseSettings):
    """Clustering configuration."""

    model_config = SettingsConfigDict(env_prefix="CLUSTERING_")

    method: str = Field(default="dpmeans", description="Clustering method: dpmeans, hdbscan, kmeans")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence embedding model"
    )
    dpmeans_lambda: Optional[float] = Field(None, description="DP-Means threshold (auto if None)")
    n_clusters: Optional[int] = Field(None, description="Number of clusters for k-means")
    min_cluster_size: int = Field(default=5, description="Minimum cluster size for HDBSCAN")


class NetworkConfig(BaseSettings):
    """Network construction configuration."""

    model_config = SettingsConfigDict(env_prefix="NETWORK_")

    min_edge_weight: int = Field(default=2, description="Minimum edge weight to include")
    top_n_nodes: Optional[int] = Field(None, description="Keep only top N nodes by degree")
    enable_filtering: bool = Field(default=True, description="Enable network filtering")


class VizConfig(BaseSettings):
    """Visualization configuration."""

    model_config = SettingsConfigDict(env_prefix="VIZ_")

    output_dir: Path = Field(default=Path("outputs"), description="Output directory")
    dpi: int = Field(default=300, description="DPI for static images")
    figure_width: float = Field(default=16.0, description="Figure width in inches")
    figure_height: float = Field(default=12.0, description="Figure height in inches")
    font_size: int = Field(default=10, description="Base font size")


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    srl: SRLConfig = Field(default_factory=SRLConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    viz: VizConfig = Field(default_factory=VizConfig)

    # General settings
    run_id: Optional[str] = Field(None, description="Run identifier")
    output_dir: Path = Field(default=Path("runs"), description="Base output directory")
    log_level: str = Field(default="INFO", description="Logging level")
    seed: int = Field(default=42, description="Random seed")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested configs
        if "llm" in data:
            data["llm"] = LLMConfig(**data["llm"])
        if "pipeline" in data:
            data["pipeline"] = PipelineConfig(**data["pipeline"])
        if "srl" in data:
            data["srl"] = SRLConfig(**data["srl"])
        if "clustering" in data:
            data["clustering"] = ClusteringConfig(**data["clustering"])
        if "network" in data:
            data["network"] = NetworkConfig(**data["network"])
        if "viz" in data:
            data["viz"] = VizConfig(**data["viz"])

        return cls(**data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        data = self.model_dump()
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_run_dir(self) -> Path:
        """Get the run directory path.

        Returns:
            Path to run directory
        """
        if self.run_id:
            run_dir = self.output_dir / self.run_id
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.output_dir / f"run_{timestamp}"

        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def load_api_keys() -> List[str]:
    """Load API keys from environment or file.

    Checks the following sources in order:
    1. SILICONFLOW_API_KEYS environment variable (comma-separated)
    2. keys.txt file (one key per line)
    3. .env file

    Returns:
        List of API keys

    Raises:
        ValueError: If no API keys are found
    """
    keys = []

    # Try environment variable
    env_keys = os.getenv("SILICONFLOW_API_KEYS", "")
    if env_keys:
        keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])

    # Try keys.txt file
    keys_file = Path("keys.txt")
    if keys_file.exists():
        with open(keys_file, "r") as f:
            file_keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            keys.extend(file_keys)

    # Try .env file (already handled by pydantic-settings, but check single key)
    if not keys:
        single_key = os.getenv("SILICONFLOW_API_KEY", "")
        if single_key:
            keys.append(single_key)

    if not keys:
        raise ValueError(
            "No API keys found. Please set SILICONFLOW_API_KEYS environment variable, "
            "create a keys.txt file, or set SILICONFLOW_API_KEY in .env file."
        )

    return keys


def get_default_config() -> Config:
    """Get default configuration with API keys loaded.

    Returns:
        Config instance with API keys
    """
    config = Config()

    # Load API keys
    try:
        api_keys = load_api_keys()
        config.llm.api_keys = api_keys
        # Limit max_workers to number of keys if not explicitly set
        if config.llm.max_workers > len(api_keys):
            config.llm.max_workers = len(api_keys)
    except ValueError as e:
        # No keys found - will fail later if LLM is actually used
        pass

    return config
