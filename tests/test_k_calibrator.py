"""
Tests for K calibration (bootstrap sampling).
"""

import pytest
from core import find_optimal_K


def test_find_optimal_k_basic():
    """Test basic K calibration."""
    # Create 100 documents with 1000 tokens each
    doc_lengths = [1000] * 100

    K = find_optimal_K(
        doc_token_counts=doc_lengths,
        T1=50_000,  # Should fit 50 documents
        T2=200_000,  # Should fit 200 documents
        target_spill_rate=0.05,
        n_bootstrap=1000,
        verbose=False
    )

    # K should be reasonable
    assert K >= 10
    assert K <= 100


def test_find_optimal_k_variable_sizes():
    """Test K calibration with variable document sizes."""
    # Mix of small and large documents
    doc_lengths = [500] * 50 + [2000] * 50

    K = find_optimal_K(
        doc_token_counts=doc_lengths,
        T1=50_000,
        T2=200_000,
        target_spill_rate=0.05,
        n_bootstrap=1000,
        verbose=False
    )

    # Should handle variability
    assert K > 0
    assert isinstance(K, int)


def test_find_optimal_k_small_corpus(small_corpus):
    """Test K calibration with actual small corpus."""
    doc_lengths = [doc.token_count for doc in small_corpus]

    K = find_optimal_K(
        doc_token_counts=doc_lengths,
        T1=20_000,
        T2=100_000,
        target_spill_rate=0.05,
        n_bootstrap=100,  # Fewer bootstrap samples for speed
        verbose=False
    )

    # Should get a valid K
    assert K > 0
    assert K <= len(small_corpus)
