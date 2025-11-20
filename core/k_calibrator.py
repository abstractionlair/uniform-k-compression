#!/usr/bin/env python3
"""
K calibration using bootstrap sampling.

Implements methodology §4.2: empirical calibration to find the largest K
where the spill rate (fraction of instances exceeding T1) is below target.
"""

import numpy as np
from typing import List, Tuple


def find_optimal_K(
    doc_token_counts: List[int],
    T1: int,
    T2: int,
    target_spill_rate: float = 0.05,
    n_bootstrap: int = 10_000,
    verbose: bool = False
) -> int:
    """
    Find optimal K via empirical bootstrap sampling.

    Uses bootstrap to empirically estimate the spill rate for each K value,
    returning the largest K where spill rate ≤ target.

    Args:
        doc_token_counts: Token counts for all documents in the layer
        T1: Small context budget (prefer to stay under this)
        T2: Large context budget (hard limit)
        target_spill_rate: Maximum acceptable fraction exceeding T1
        n_bootstrap: Number of bootstrap samples to run
        verbose: Print calibration progress

    Returns:
        Optimal K (largest K with spill_rate ≤ target)

    Raises:
        ValueError: If no valid K exists (even K=1 violates constraints)
    """
    lengths = np.array(doc_token_counts)
    N = len(lengths)

    if N == 0:
        raise ValueError("doc_token_counts cannot be empty")

    # Compute K_max: largest K such that K largest documents fit in T2
    sorted_lengths = np.sort(lengths)[::-1]  # Descending
    cumsum = np.cumsum(sorted_lengths)
    K_max = np.searchsorted(cumsum, T2, side='right')

    if K_max == 0:
        raise ValueError(f"Even single largest document ({sorted_lengths[0]} tokens) exceeds T2 ({T2})")

    if verbose:
        print(f"K calibration: N={N} documents")
        print(f"  Document lengths: min={lengths.min()}, mean={lengths.mean():.0f}, max={lengths.max()}")
        print(f"  K_max={K_max} (safety constraint from T2)")

    # Try K values from large to small
    for K in range(K_max, 0, -1):
        if K > N:
            continue  # Can't sample more documents than we have

        spill_count = 0

        # Bootstrap: repeatedly sample K documents and check if sum > T1
        for _ in range(n_bootstrap):
            sample = np.random.choice(lengths, size=K, replace=False)
            if sample.sum() > T1:
                spill_count += 1

        actual_spill_rate = spill_count / n_bootstrap

        if verbose and K % max(1, K_max // 10) == 0:
            print(f"  K={K:3d}: spill_rate={actual_spill_rate:.3f}")

        if actual_spill_rate <= target_spill_rate:
            if verbose:
                print(f"  ✓ Optimal K={K} (spill_rate={actual_spill_rate:.3f} ≤ {target_spill_rate})")
            return K

    # Fallback: even K=1 exceeds target spill rate
    # This can happen with very heterogeneous distributions
    if verbose:
        print(f"  ⚠️  Warning: Even K=1 exceeds target spill rate")
        print(f"     Returning K=1 anyway (required for progress)")

    return 1


def estimate_layer_stats(
    doc_token_counts: List[int],
    K: int,
    T1: int,
    n_samples: int = 1000
) -> Tuple[float, float, float]:
    """
    Estimate statistics for a layer with given K.

    Useful for predicting behavior before running the layer.

    Args:
        doc_token_counts: Token counts for documents
        K: Documents per instance
        T1: Small context budget
        n_samples: Number of samples for estimation

    Returns:
        Tuple of (mean_instance_tokens, spill_rate, utilization)
            - mean_instance_tokens: Average tokens per instance
            - spill_rate: Fraction exceeding T1
            - utilization: mean_instance_tokens / T1
    """
    lengths = np.array(doc_token_counts)
    N = len(lengths)

    if K > N:
        K = N  # Can't sample more than we have

    instance_sizes = []
    spills = 0

    for _ in range(n_samples):
        sample = np.random.choice(lengths, size=K, replace=False)
        total = sample.sum()
        instance_sizes.append(total)
        if total > T1:
            spills += 1

    mean_size = np.mean(instance_sizes)
    spill_rate = spills / n_samples
    utilization = mean_size / T1

    return mean_size, spill_rate, utilization


if __name__ == "__main__":
    # Test on synthetic data
    print("="*70)
    print("K CALIBRATION TESTS")
    print("="*70)

    # Test 1: Uniform distribution
    print("\nTest 1: Uniform distribution (3k-5k tokens)")
    uniform_docs = [np.random.randint(3000, 5000) for _ in range(100)]
    K = find_optimal_K(uniform_docs, T1=40_000, T2=200_000, verbose=True)
    mean, spill, util = estimate_layer_stats(uniform_docs, K, T1=40_000)
    print(f"  Predicted: mean={mean:.0f} tokens, spill={spill:.3f}, utilization={util:.2f}")

    # Test 2: Heavy-tailed distribution
    print("\nTest 2: Heavy-tailed distribution")
    # 80% small docs (1-3k), 20% large docs (10-30k)
    heavy_tail = (
        [np.random.randint(1000, 3000) for _ in range(80)] +
        [np.random.randint(10_000, 30_000) for _ in range(20)]
    )
    np.random.shuffle(heavy_tail)
    K = find_optimal_K(heavy_tail, T1=40_000, T2=200_000, verbose=True)
    mean, spill, util = estimate_layer_stats(heavy_tail, K, T1=40_000)
    print(f"  Predicted: mean={mean:.0f} tokens, spill={spill:.3f}, utilization={util:.2f}")

    # Test 3: Typical Layer 1 (from methodology doc example)
    print("\nTest 3: Layer 1 simulation (4,452 docs, mean ~4k tokens)")
    layer1_docs = [np.random.gamma(2, 2000) for _ in range(4452)]  # Gamma dist, mean ~4k
    K = find_optimal_K(layer1_docs, T1=154_000, T2=769_000, verbose=True)
    mean, spill, util = estimate_layer_stats(layer1_docs, K, T1=154_000)
    print(f"  Predicted: mean={mean:.0f} tokens, spill={spill:.3f}, utilization={util:.2f}")

    print("\n" + "="*70)
    print("Tests complete!")
