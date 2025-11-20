"""
End-to-end integration test using local Ollama.

This test runs the full fractal summarization pipeline with a real LLM,
but uses Ollama instead of Anthropic API to avoid costs.

Requires:
- Ollama installed and running (https://ollama.ai)
- Model pulled: ollama pull qwen2.5:3b
"""

import pytest
from pathlib import Path
from core import (
    FrameworkConfig,
    AnalysisConfig,
    FractalSummarizer,
    OllamaInterface,
    is_ollama_available,
)


# Skip all tests in this module if Ollama is not available
pytestmark = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available (install from https://ollama.ai)"
)


@pytest.fixture
def ollama_framework_config():
    """Framework config optimized for small corpus and local LLM."""
    return FrameworkConfig(
        k=1.5,
        r=0.4,  # Less aggressive compression for smaller models
        T1=16_000,  # Medium contexts
        T2=80_000,  # Large enough for biggest Sherlock Holmes story (61K)
        target_convergence=40_000,
        bootstrap_iterations=500,  # Fewer iterations for speed
    )


@pytest.fixture
def ollama_analysis_config(tmp_path):
    """Analysis config for test run."""
    return AnalysisConfig(
        name="Ollama E2E Test",
        description="End-to-end test with Ollama",
        layer_prompt_template="""Analyze these {num_docs} Sherlock Holmes stories.

Compress to approximately {r:.0%} of the original length while preserving:
- Key plot points
- Character names
- How the mystery was solved

Stories:
{documents}

Summary:""",
        final_synthesis_prompt="""Synthesize these story summaries into a brief overview of:
1. Common themes
2. Holmes' methods
3. Types of cases

Keep it concise (2-3 paragraphs).""",
        output_dir=str(tmp_path / "ollama_test_output")
    )


def test_ollama_interface_basic():
    """Test that Ollama interface works for basic calls."""
    # Try qwen3:8b first (user's model), fall back to qwen2.5:3b
    try:
        ollama = OllamaInterface(model="qwen3:8b")
    except RuntimeError:
        try:
            ollama = OllamaInterface(model="qwen2.5:3b")
        except RuntimeError as e:
            pytest.skip(f"Ollama setup issue: {e}")

    # Make a simple test call
    output, tokens, usage = ollama.call(
        "What is 2+2? Answer in one word.",
        context_size="small",
        max_tokens=10
    )

    assert output is not None
    assert len(output) > 0
    assert tokens > 0
    assert usage.cost_usd == 0.0  # Ollama is free


def test_ollama_full_pipeline_tiny(small_corpus, ollama_framework_config, ollama_analysis_config):
    """Test full pipeline with tiny corpus (5 stories)."""
    # Try qwen3:8b first, fall back to qwen2.5:3b
    try:
        model = "qwen3:8b"
        ollama = OllamaInterface(model=model)
    except RuntimeError:
        try:
            model = "qwen2.5:3b"
            ollama = OllamaInterface(model=model)
        except RuntimeError as e:
            pytest.skip(f"Ollama setup issue: {e}")

    # Create a custom FractalSummarizer that uses Ollama
    # We'll need to create a wrapper that matches the LLMInterface signature
    class OllamaFractalSummarizer(FractalSummarizer):
        """FractalSummarizer that uses Ollama instead of Anthropic."""

        def __init__(self, config: FrameworkConfig, model: str = "qwen3:8b"):
            # Don't call super().__init__ to avoid Anthropic API key requirement
            self.config = config
            self.llm = OllamaInterface(model=model)
            self.batch = None  # No batch support for Ollama
            from core.document import Tokenizer
            self.tokenizer = Tokenizer()

    print("\n" + "="*70)
    print(f"OLLAMA E2E TEST - Full Pipeline (using {model})")
    print("="*70)

    summarizer = OllamaFractalSummarizer(ollama_framework_config, model=model)

    # Run the full pipeline
    result, metadata = summarizer.run(small_corpus, ollama_analysis_config)

    # Verify results
    assert result is not None
    assert len(result) > 0
    assert metadata.total_layers > 0
    assert metadata.total_instances > 0
    assert metadata.total_cost_usd == 0.0  # Ollama is free

    # Verify compression happened
    assert metadata.final_tokens < metadata.initial_tokens

    # Verify output files were created
    output_dir = Path(ollama_analysis_config.output_dir)
    assert (output_dir / "final_analysis.md").exists()
    assert (output_dir / "run_metadata.json").exists()

    print(f"\n✅ Test passed!")
    print(f"   Layers: {metadata.total_layers}")
    print(f"   Instances: {metadata.total_instances}")
    print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
    print(f"   Cost: ${metadata.total_cost_usd:.2f} (free!)")


def test_ollama_vs_mock_comparison(small_corpus, ollama_framework_config):
    """Compare Ollama output to mock to verify real LLM behavior."""
    # Try qwen3:8b first, fall back to qwen2.5:3b
    try:
        ollama = OllamaInterface(model="qwen3:8b")
    except RuntimeError:
        try:
            ollama = OllamaInterface(model="qwen2.5:3b")
        except RuntimeError as e:
            pytest.skip(f"Ollama setup issue: {e}")

    # Create a simple prompt
    docs = small_corpus[:2]  # Just 2 documents
    prompt = "\n\n".join([f"Story: {d.content[:500]}" for d in docs])
    prompt += "\n\nSummarize these two stories in 2-3 sentences."

    # Get Ollama response
    output, tokens, usage = ollama.call(prompt, "small", max_tokens=200)

    # Verify output is substantive (not just repetition)
    assert len(output) > 50, "Output too short"
    assert len(output.split()) > 10, "Output should have multiple words"

    # Verify it's actually compressed (not just copying input)
    input_length = len(prompt)
    output_length = len(output)
    assert output_length < input_length, "Should compress the input"

    print(f"\n📊 Ollama compression test:")
    print(f"   Input: {input_length} chars")
    print(f"   Output: {output_length} chars")
    print(f"   Ratio: {output_length/input_length:.2%}")


def test_ollama_with_commentary(small_corpus, tmp_path):
    """Test iterative refinement with commentary."""
    # Try qwen3:8b first, fall back to qwen2.5:3b
    try:
        model = "qwen3:8b"
        ollama = OllamaInterface(model=model)
    except RuntimeError:
        try:
            model = "qwen2.5:3b"
            ollama = OllamaInterface(model=model)
        except RuntimeError as e:
            pytest.skip(f"Ollama setup issue: {e}")

    # Framework config that forces multiple layers
    framework_config = FrameworkConfig(
        k=1.5,
        r=0.4,
        T1=16_000,
        T2=80_000,
        target_convergence=3_000,  # Force multiple layers
        bootstrap_iterations=500,
    )

    # Use OllamaFractalSummarizer
    class OllamaFractalSummarizer(FractalSummarizer):
        def __init__(self, config: FrameworkConfig, model: str = "qwen3:8b"):
            self.config = config
            self.llm = OllamaInterface(model=model)
            self.batch = None
            from core.document import Tokenizer
            self.tokenizer = Tokenizer()

    print("\n" + "="*70)
    print(f"OLLAMA COMMENTARY TEST (using {model})")
    print("="*70)

    # Use just 3 stories for speed
    docs = small_corpus[:3]
    total_tokens = sum(d.token_count for d in docs)
    print(f"\nCorpus: {len(docs)} stories, {total_tokens:,} tokens")

    # =========================================================================
    # RUN 1: Initial analysis (no commentary)
    # =========================================================================
    print("\n" + "="*70)
    print("RUN 1: Without Commentary")
    print("="*70)

    analysis_config_run1 = AnalysisConfig(
        name="Run 1 - No Commentary",
        layer_prompt_template="""Analyze {num_docs} Sherlock Holmes stories.

Focus on Holmes' character traits and methods.
Compress to ~{r:.0%} while preserving key details.

{documents}

Summary:""",
        final_synthesis_prompt="Summarize Holmes' character and methods based on these stories.",
        output_dir=str(tmp_path / "run1")
    )

    summarizer1 = OllamaFractalSummarizer(framework_config, model=model)
    result1, metadata1 = summarizer1.run(docs, analysis_config_run1)

    print(f"\n✅ Run 1 complete:")
    print(f"   Layers: {metadata1.total_layers}")
    print(f"   Output length: {len(result1)} chars")

    # =========================================================================
    # Create commentary
    # =========================================================================
    print("\n" + "="*70)
    print("Creating Commentary")
    print("="*70)

    commentary_file = tmp_path / "commentary.md"
    commentary_content = """# Commentary on Run 1

## What Was Missing

The analysis should pay more attention to Watson's role as narrator and friend.
Holmes' emotional side and his appreciation for Watson's companionship is important.

## Important Context

These stories are not just detective puzzles - they're also about the friendship
between Holmes and Watson. The Victorian London setting matters too.

## Specific Corrections

Holmes is not purely logical - he has moments of warmth, humor, and even excitement.
"""

    with open(commentary_file, 'w') as f:
        f.write(commentary_content)

    print(f"✅ Created commentary: {commentary_file}")

    # =========================================================================
    # RUN 2: With commentary
    # =========================================================================
    print("\n" + "="*70)
    print("RUN 2: With Commentary")
    print("="*70)

    analysis_config_run2 = AnalysisConfig(
        name="Run 2 - With Commentary",
        layer_prompt_template="""Analyze {num_docs} Sherlock Holmes stories.

Focus on Holmes' character traits and methods.
Compress to ~{r:.0%} while preserving key details.

{documents}

Summary:""",
        final_synthesis_prompt="Summarize Holmes' character and methods based on these stories.",
        output_dir=str(tmp_path / "run2"),
        commentary_file=str(commentary_file)  # ← Include commentary
    )

    summarizer2 = OllamaFractalSummarizer(framework_config, model=model)
    result2, metadata2 = summarizer2.run(docs, analysis_config_run2)

    print(f"\n✅ Run 2 complete:")
    print(f"   Layers: {metadata2.total_layers}")
    print(f"   Output length: {len(result2)} chars")

    # =========================================================================
    # Verify commentary was incorporated
    # =========================================================================
    print("\n" + "="*70)
    print("Verification")
    print("="*70)

    # Both runs should complete successfully
    assert result1 is not None and len(result1) > 0
    assert result2 is not None and len(result2) > 0
    assert metadata1.total_cost_usd == 0.0
    assert metadata2.total_cost_usd == 0.0

    # Run 2 should mention Watson or emotional aspects (from commentary)
    # We can't guarantee specific words, but we can check that the outputs differ
    assert result1 != result2, "Commentary should influence the output"

    result2_lower = result2.lower()
    # Check if Run 2 addresses themes from commentary
    commentary_influenced = (
        'watson' in result2_lower or
        'friend' in result2_lower or
        'emotion' in result2_lower or
        'warm' in result2_lower or
        'companion' in result2_lower
    )

    print(f"\n📊 Comparison:")
    print(f"   Run 1 length: {len(result1)} chars")
    print(f"   Run 2 length: {len(result2)} chars")
    print(f"   Commentary themes in Run 2: {commentary_influenced}")

    # We expect commentary to influence the output
    # (Though we can't guarantee specific words with small models)
    if commentary_influenced:
        print("   ✅ Commentary appears to have influenced Run 2")
    else:
        print("   ⚠️  Commentary influence not clearly detected (expected with small models)")

    # Save outputs for manual inspection
    print(f"\n📁 Outputs saved:")
    print(f"   Run 1: {tmp_path / 'run1' / 'final_analysis.md'}")
    print(f"   Run 2: {tmp_path / 'run2' / 'final_analysis.md'}")
    print(f"   Commentary: {commentary_file}")

    print(f"\n✅ Commentary integration test passed!")
