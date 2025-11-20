"""
Pytest configuration and fixtures for fractal summarization tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import Tokenizer


@pytest.fixture
def test_data_dir():
    """Path to test data directory containing Sherlock Holmes stories."""
    return Path(__file__).parent / "test_data" / "stories"


@pytest.fixture
def tokenizer():
    """Shared tokenizer instance."""
    return Tokenizer()


@pytest.fixture
def sample_documents(test_data_dir, tokenizer):
    """Load 10 sample documents for testing."""
    from utilities import load_documents

    docs = load_documents(
        directory_path=str(test_data_dir),
        pattern="*.txt",
        limit=10
    )
    return docs


@pytest.fixture
def small_corpus(test_data_dir, tokenizer):
    """Load 5 small documents for quick tests."""
    from utilities import load_documents

    docs = load_documents(
        directory_path=str(test_data_dir),
        pattern="*.txt",
        limit=5
    )
    return docs
