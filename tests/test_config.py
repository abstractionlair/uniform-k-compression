"""
Tests for configuration classes.
"""

import pytest
from core import FrameworkConfig, AnalysisConfig, PreprocessingConfig


def test_framework_config_defaults():
    """Test framework config with default parameters."""
    config = FrameworkConfig()

    assert config.k == 1.5
    assert config.r == 0.3
    assert config.T1 == 154_000
    assert config.T2 == 769_000
    assert config.target_convergence == 700_000


def test_framework_config_alpha():
    """Test alpha calculation (effective layer compression)."""
    config = FrameworkConfig(k=1.5, r=0.3)
    assert config.alpha == pytest.approx(0.45, rel=0.01)


def test_framework_config_validation():
    """Test that invalid configs raise errors."""
    with pytest.raises(ValueError, match="k should be"):
        FrameworkConfig(k=5.0)

    with pytest.raises(ValueError, match="r should be"):
        FrameworkConfig(r=0.8)

    with pytest.raises(ValueError, match="T1.*must be < T2"):
        FrameworkConfig(T1=800_000, T2=200_000)


def test_analysis_config_creation():
    """Test analysis config creation."""
    config = AnalysisConfig(
        name="Test Analysis",
        layer_prompt_template="Analyze {num_docs} documents",
        final_synthesis_prompt="Synthesize findings",
        output_dir="output/test"
    )

    assert config.name == "Test Analysis"
    assert "{num_docs}" in config.layer_prompt_template
    assert config.output_dir == "output/test"


def test_preprocessing_config_defaults():
    """Test preprocessing config defaults."""
    config = PreprocessingConfig()

    assert config.chunking_threshold == 50_000
    assert config.target_chunk_size == 40_000
    assert config.tokenizer == "cl100k_base"
