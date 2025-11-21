#!/usr/bin/env python3
"""
Configuration management for uniform-K fractal summarization.

Separates framework configuration (k, r, context budgets) from
analysis configuration (prompts, output paths).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def resolve_model_name(model: str) -> str:
    """
    Resolve simplified model names to full API names.

    The Claude CLI accepts 'sonnet' and 'sonnet[1m]', but the API
    always uses the base model name. Extended context is accessed via
    beta flag, not a different model name.

    Args:
        model: Model name (can be simplified or full)

    Returns:
        Full model name for API
    """
    # Map simplified names to full names
    model_map = {
        'sonnet': 'claude-sonnet-4-5-20250929',
        'sonnet[1m]': 'claude-sonnet-4-5-20250929',  # Same model, use beta flag for 1M
        'opus': 'claude-opus-4-20250514',
        'haiku': 'claude-3-5-haiku-20241022'
    }

    # Return mapped name if found, otherwise assume it's already full
    return model_map.get(model, model)


@dataclass
class FrameworkConfig:
    """
    Core framework parameters for uniform-K summarization.

    These control the algorithm behavior and resource usage.
    """
    # Sampling density (expected times each token is read per layer)
    k: float = 1.5

    # Compression ratio per instance (output/input)
    r: float = 0.3

    # Small context budget (tokens)
    T1: int = 154_000

    # Large context budget (tokens)
    T2: int = 769_000

    # Target convergence (stop when total < this)
    # Should be less than T2 to fit in final synthesis
    target_convergence: int = 700_000

    # Provider selection ('anthropic', 'openai', 'google', 'xai')
    provider: str = "anthropic"

    # Model to use for small contexts (200K)
    model: str = "sonnet"

    # Model to use for large contexts (1M)
    large_context_model: str = "sonnet[1m]"

    # Temperature for sampling
    temperature: float = 1.0

    # Target spill rate for K calibration
    target_spill_rate: float = 0.05

    # Bootstrap iterations for K calibration
    bootstrap_iterations: int = 10_000

    # Batch API mode (50% cost, takes hours instead of minutes)
    use_batch_api: bool = False
    batch_poll_interval: int = 300  # Seconds between status checks

    def __post_init__(self):
        """Validate configuration."""
        if not 0.5 <= self.k <= 3.0:
            raise ValueError(f"k should be in [0.5, 3.0], got {self.k}")
        if not 0.1 <= self.r <= 0.5:
            raise ValueError(f"r should be in [0.1, 0.5], got {self.r}")
        if self.T1 >= self.T2:
            raise ValueError(f"T1 ({self.T1}) must be < T2 ({self.T2})")
        if self.target_convergence >= self.T2:
            raise ValueError(f"target_convergence ({self.target_convergence}) should be < T2 ({self.T2})")
        if not 0.0 < self.target_spill_rate < 0.5:
            raise ValueError(f"target_spill_rate should be in (0, 0.5), got {self.target_spill_rate}")

        # Validate provider
        valid_providers = ['anthropic', 'openai', 'google', 'gemini', 'xai', 'grok']
        if self.provider.lower() not in valid_providers:
            raise ValueError(
                f"Invalid provider: {self.provider}. "
                f"Must be one of: {', '.join(valid_providers)}"
            )

        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature should be in [0.0, 2.0], got {self.temperature}")

    @property
    def alpha(self) -> float:
        """Expected layer compression ratio: α ≈ k·r"""
        return self.k * self.r

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'FrameworkConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, path: Path) -> 'FrameworkConfig':
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class AnalysisConfig:
    """
    Analysis-specific configuration.

    These define the prompts and output for a specific analysis task.
    """
    # Analysis name
    name: str

    # Template for layer prompts
    # Available variables: {documents}, {layer_num}, {k}, {r}, {num_docs}
    layer_prompt_template: str

    # Prompt for final synthesis
    final_synthesis_prompt: str

    # Output directory
    output_dir: str

    # Optional description
    description: str = ""

    # Optional commentary file for iterative refinement
    # Path to markdown file with user feedback from previous run
    commentary_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'AnalysisConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, path: Path) -> 'AnalysisConfig':
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing utilities (optional).

    These control document preparation steps like chunking.
    """
    # Threshold for chunking (tokens)
    chunking_threshold: int = 50_000

    # Target chunk size (tokens)
    target_chunk_size: int = 40_000

    # Tokenizer to use
    tokenizer: str = "cl100k_base"  # tiktoken model name

    # Chunk overlap (tokens)
    chunk_overlap: int = 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PreprocessingConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, path: Path) -> 'PreprocessingConfig':
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Example usage
    framework = FrameworkConfig(k=1.5, r=0.3)
    print("Framework Config:")
    print(f"  k={framework.k}, r={framework.r}")
    print(f"  α≈{framework.alpha:.2f}")
    print(f"  T1={framework.T1:,}, T2={framework.T2:,}")

    analysis = AnalysisConfig(
        name="Test Analysis",
        layer_prompt_template="Analyze these {num_docs} documents...",
        final_synthesis_prompt="Synthesize findings...",
        output_dir="output/test"
    )
    print(f"\nAnalysis: {analysis.name}")
    print(f"  Output: {analysis.output_dir}")

    prep = PreprocessingConfig()
    print("\nPreprocessing:")
    print(f"  Chunk threshold: {prep.chunking_threshold:,} tokens")
