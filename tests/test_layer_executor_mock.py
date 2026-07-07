"""
Tests for layer execution with mocked LLM.

These tests verify the layer execution logic without making actual API calls.
"""

from core.layer_executor import run_layer


class MockLLM:
    """Mock LLM that returns compressed versions of input."""

    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
        self.calls = []

    def __call__(self, prompt: str, context_size: str):
        """Mock LLM call that compresses input deterministically."""
        self.calls.append({"prompt": prompt, "context_size": context_size})

        # Simple compression: return first N% of input
        input_lines = prompt.split("\n")
        output_lines_count = max(1, int(len(input_lines) * self.compression_ratio))
        output = "\n".join(input_lines[:output_lines_count])

        # Estimate output tokens (rough approximation)
        output_tokens = len(output) // 4

        return output, output_tokens


def test_run_layer_basic(small_corpus):
    """Test basic layer execution with mock LLM."""
    mock_llm = MockLLM(compression_ratio=0.3)

    def mock_prompt_builder(docs, layer_num, k, r):
        # Simple prompt that concatenates documents
        return "\n\n".join([d.content for d in docs])

    output_docs, stats = run_layer(
        documents=small_corpus,
        k=1.5,
        K=3,  # Sample 3 documents at a time
        r=0.3,
        T1=50_000,
        T2=200_000,
        layer_num=1,
        llm_caller=mock_llm,
        prompt_builder=mock_prompt_builder,
        seed=42,
    )

    # Should create n = round(k*N/K) instances (rounded half-up, not floored)
    expected_instances = int(1.5 * len(small_corpus) / 3 + 0.5)
    assert len(output_docs) == expected_instances
    assert stats.n_instances == expected_instances
    assert stats.layer_num == 1
    assert stats.K == 3

    # Should have made LLM calls
    assert len(mock_llm.calls) == expected_instances

    # Output documents should have content
    assert all(doc.content for doc in output_docs)
    assert all(doc.token_count > 0 for doc in output_docs)


def test_run_layer_compression(small_corpus):
    """Test that layer achieves expected compression."""
    mock_llm = MockLLM(compression_ratio=0.3)

    def mock_prompt_builder(docs, layer_num, k, r):
        return "\n\n".join([d.content for d in docs])

    corpus_tokens = sum(d.token_count for d in small_corpus)

    output_docs, stats = run_layer(
        documents=small_corpus,
        k=1.5,
        K=3,
        r=0.3,
        T1=50_000,
        T2=200_000,
        layer_num=1,
        llm_caller=mock_llm,
        prompt_builder=mock_prompt_builder,
        seed=42,
    )

    output_tokens = sum(d.token_count for d in output_docs)

    # Should have reasonable compression
    # Output should be smaller than corpus
    assert output_tokens < corpus_tokens

    # Note: stats.total_input_tokens counts LLM input (which can exceed corpus
    # size when documents are sampled multiple times)
    assert stats.total_input_tokens >= corpus_tokens  # Can be larger due to resampling
    assert stats.compression_ratio > 0


def test_run_layer_deterministic(small_corpus):
    """Test that layer execution is deterministic with same seed."""
    mock_llm1 = MockLLM(compression_ratio=0.3)
    mock_llm2 = MockLLM(compression_ratio=0.3)

    def mock_prompt_builder(docs, layer_num, k, r):
        return "\n\n".join([d.content for d in docs])

    # Run twice with same seed
    output1, stats1 = run_layer(
        documents=small_corpus,
        k=1.5,
        K=3,
        r=0.3,
        T1=50_000,
        T2=200_000,
        layer_num=1,
        llm_caller=mock_llm1,
        prompt_builder=mock_prompt_builder,
        seed=42,
    )

    output2, stats2 = run_layer(
        documents=small_corpus,
        k=1.5,
        K=3,
        r=0.3,
        T1=50_000,
        T2=200_000,
        layer_num=1,
        llm_caller=mock_llm2,
        prompt_builder=mock_prompt_builder,
        seed=42,
    )

    # Should produce same results
    assert len(output1) == len(output2)
    assert stats1.n_instances == stats2.n_instances

    # Document IDs should match (same sampling order)
    for d1, d2 in zip(output1, output2):
        assert d1.doc_id == d2.doc_id


def test_run_layer_sampling_density(small_corpus):
    """Test that k parameter affects sampling density correctly."""
    mock_llm = MockLLM(compression_ratio=0.3)

    def mock_prompt_builder(docs, layer_num, k, r):
        return "\n\n".join([d.content for d in docs])

    # With k=1.5, each document should appear in ~1.5 instances
    output, stats = run_layer(
        documents=small_corpus,
        k=1.5,
        K=2,
        r=0.3,
        T1=50_000,
        T2=200_000,
        layer_num=1,
        llm_caller=mock_llm,
        prompt_builder=mock_prompt_builder,
        seed=42,
    )

    # n = round(k * N / K), rounded half-up
    expected_n = int(1.5 * len(small_corpus) / 2 + 0.5)
    assert stats.n_instances == expected_n


def test_run_layer_instance_count_rounds_to_nearest(small_corpus):
    """n_instances must round k*N/K to the nearest integer, not floor it.

    Flooring systematically under-sampled small layers relative to the
    documented "each document is read k times on average" invariant:
    with N=5, K=3, k=1.5, floor gave 2 instances (1.2 expected reads);
    rounding gives 3 instances (1.8 expected reads, the closest integer
    realization of the requested k=1.5).
    """
    mock_llm = MockLLM(compression_ratio=0.3)

    def mock_prompt_builder(docs, layer_num, k, r):
        return "\n\n".join([d.content for d in docs])

    N = len(small_corpus)
    assert N == 5  # Fixture assumption for the arithmetic below

    k, K = 1.5, 3
    _, stats = run_layer(
        documents=small_corpus,
        k=k,
        K=K,
        r=0.3,
        T1=50_000,
        T2=200_000,
        layer_num=1,
        llm_caller=mock_llm,
        prompt_builder=mock_prompt_builder,
        seed=42,
    )

    assert stats.n_instances == 3  # round(2.5) half-up, not floor -> 2

    # Rounded n is the integer that best tracks the requested k
    realized_reads = stats.n_instances * K / N
    floored_reads = int(k * N / K) * K / N
    assert abs(realized_reads - k) <= abs(floored_reads - k)
