"""
Focused test: Use Ollama to answer a specific question from a curated corpus.

This demonstrates using fractal summarization to extract specific insights
from multiple documents, answering: "What are Holmes' deductive methods?"

Hand-picked stories:
- A Scandal in Bohemia (Irene Adler)
- The Red-Headed League (pawnbroker mystery)
- The Adventure of the Blue Carbuncle (Christmas goose)
- The Adventure of the Speckled Band (locked room mystery)
"""

from pathlib import Path

import pytest

from core import (
    AnalysisConfig,
    FractalSummarizer,
    FrameworkConfig,
    OllamaInterface,
    is_ollama_available,
)

pytestmark = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available"
)


class OllamaFractalSummarizer(FractalSummarizer):
    """FractalSummarizer using Ollama."""

    def __init__(self, config: FrameworkConfig, model: str = "qwen3:8b"):
        self.config = config
        self.llm = OllamaInterface(model=model)
        self.batch = None
        from core.document import Tokenizer
        self.tokenizer = Tokenizer()


def test_holmes_deductive_methods(tmp_path):
    """
    Answer specific question: What are Holmes' deductive methods?

    Uses 4 hand-picked stories, forces multiple layers to test
    full pipeline behavior.
    """
    test_data_dir = Path(__file__).parent / "test_data" / "stories"

    # Hand-pick 4 classic stories
    story_files = [
        "03_a_scandal_in_bohemia.txt",
        "04_the_red_headed_league.txt",
        "09_the_adventure_of_the_blue_carbuncle.txt",
        "10_the_adventure_of_the_speckled_band.txt",
    ]

    print("\n" + "="*70)
    print("FOCUSED TEST: Analyzing Holmes' Deductive Methods")
    print("="*70)

    # Load specific stories
    from core import Tokenizer
    tokenizer = Tokenizer()

    docs = []
    for filename in story_files:
        filepath = test_data_dir / filename
        with open(filepath, 'r') as f:
            content = f.read()

        from core import create_document_with_tokens
        doc = create_document_with_tokens(
            content=content,
            doc_id=filename.replace('.txt', ''),
            tokenizer=tokenizer
        )
        docs.append(doc)
        print(f"  ✓ Loaded: {filename} ({doc.token_count:,} tokens)")

    total_tokens = sum(d.token_count for d in docs)
    print(f"\n  Total corpus: {len(docs)} stories, {total_tokens:,} tokens")

    # Configure to force multiple layers
    framework_config = FrameworkConfig(
        k=2.0,  # Higher sampling density for better coverage
        r=0.4,  # Gentle compression
        T1=16_000,
        T2=80_000,
        target_convergence=3_000,  # Force multiple layers
        bootstrap_iterations=300,
    )

    # Focused analysis on deductive methods
    analysis_config = AnalysisConfig(
        name="Holmes Deductive Methods",
        description="Extract Holmes' investigative techniques",
        layer_prompt_template="""You are analyzing {num_docs} Sherlock Holmes stories to identify his deductive methods.

QUESTION: What are Sherlock Holmes' key deductive methods and investigative techniques?

Focus on:
1. What Holmes observes (physical clues, behaviors, patterns)
2. How he draws logical inferences from observations
3. His systematic approach to investigation
4. Specific techniques he mentions or demonstrates

Stories to analyze:
{documents}

Compress to ~{r:.0%} while extracting examples of Holmes' methods.

Summary of deductive methods found:""",
        final_synthesis_prompt="""You have analyzed multiple Sherlock Holmes stories to identify his deductive methods.

Based on the summaries below, provide a comprehensive answer to:
**"What are Sherlock Holmes' key deductive methods and investigative techniques?"**

Structure your answer:
1. Core principles (how Holmes thinks)
2. Observational techniques (what he looks for)
3. Logical inference methods (how he deduces)
4. Systematic approaches (his process)
5. Specific examples from the stories

Be specific and cite examples where possible.""",
        output_dir=str(tmp_path / "holmes_methods")
    )

    print("\n  Question: What are Holmes' deductive methods?")
    print(f"  Framework: k={framework_config.k}, r={framework_config.r}")
    print(f"  Target: Multiple layers (converge at {framework_config.target_convergence:,} tokens)")

    # Run analysis
    print("\n" + "="*70)
    print("RUNNING ANALYSIS")
    print("="*70)

    try:
        summarizer = OllamaFractalSummarizer(framework_config, model="qwen3:8b")
        result, metadata = summarizer.run(docs, analysis_config)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"  Layers: {metadata.total_layers}")
        print(f"  Instances: {metadata.total_instances}")
        print(f"  Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
        print(f"  Duration: {metadata.duration_seconds/60:.1f} minutes")
        print("  Cost: $0.00 (free!)")

        # Show the answer
        print("\n" + "="*70)
        print("ANSWER: Holmes' Deductive Methods")
        print("="*70)
        print(result)
        print("="*70)

        # Save output
        output_file = tmp_path / "holmes_methods" / "final_analysis.md"
        print(f"\n  Saved to: {output_file}")

        # Verify it worked
        assert result is not None
        assert len(result) > 200  # Should have substantive answer
        assert metadata.total_layers >= 1
        assert metadata.total_cost_usd == 0.0

        # Check that output mentions deduction/methods
        result_lower = result.lower()
        assert any(word in result_lower for word in ['deduc', 'observ', 'method', 'logic'])

        print("\n✅ Test passed! Question answered successfully.")

    except RuntimeError as e:
        pytest.skip(f"Ollama issue: {e}")


if __name__ == "__main__":
    # Run standalone for manual testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_holmes_deductive_methods(Path(tmpdir))
