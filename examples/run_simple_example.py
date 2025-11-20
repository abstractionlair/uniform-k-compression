#!/usr/bin/env python3
"""
Simple example demonstrating fractal summarization on Sherlock Holmes stories.

This example requires an Anthropic API key to run.

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python examples/run_simple_example.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import FrameworkConfig, AnalysisConfig, FractalSummarizer
from utilities import load_documents


def main():
    print("=" * 70)
    print("FRACTAL SUMMARIZATION - SIMPLE EXAMPLE")
    print("=" * 70)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nPlease set your API key:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("\nThen run this script again.")
        return 1

    # Load sample documents
    print("\n📚 Loading 5 Sherlock Holmes stories from examples/sample_data/...")
    sample_data_dir = Path(__file__).parent / "sample_data"

    docs = load_documents(
        directory_path=str(sample_data_dir),
        pattern="*.txt"
    )

    print(f"   Loaded {len(docs)} documents")
    print(f"   Total tokens: {sum(d.token_count for d in docs):,}")

    # Configure framework (conservative settings for small corpus)
    framework_config = FrameworkConfig(
        k=1.5,  # Each document read 1.5 times per layer
        r=0.3,  # Compress to 30% of input size
        T1=20_000,  # Small context budget (20K tokens)
        T2=100_000,  # Large context budget (100K tokens)
        target_convergence=50_000,  # Stop when under 50K tokens
        bootstrap_iterations=1000,  # Bootstrap samples for K calibration
    )

    # Configure analysis
    analysis_config = AnalysisConfig(
        name="Sherlock Holmes Stories",
        description="Example analysis of 5 Sherlock Holmes stories",
        layer_prompt_template="""You are analyzing {num_docs} Sherlock Holmes stories at layer {layer_num}.

Your task is to compress these to approximately {r:.0%} of the original size while preserving:
1. Key plot points and mysteries
2. Important character interactions
3. Holmes' deductive methods
4. Resolution of each case

Documents:
{documents}

Provide a concise summary:""",
        final_synthesis_prompt="""You have received summaries of Sherlock Holmes stories. Create a comprehensive analysis that:

1. Identifies common themes across the stories
2. Describes Holmes' typical investigative approach
3. Notes any patterns in the types of cases
4. Highlights the most interesting cases

Keep your analysis focused and well-organized.""",
        output_dir="examples/output",
    )

    print(f"\n⚙️  Framework configuration:")
    print(f"   k={framework_config.k}, r={framework_config.r}")
    print(f"   α≈{framework_config.alpha:.2f} (effective layer compression)")
    print(f"   Context budgets: T1={framework_config.T1:,}, T2={framework_config.T2:,}")

    # Estimate cost
    total_tokens = sum(d.token_count for d in docs)
    estimated_instances = int((framework_config.k * len(docs)) / 3)  # Rough estimate
    estimated_cost = (total_tokens / 1_000_000) * 3 * estimated_instances * 0.5
    print(f"\n💰 Estimated cost: ~${estimated_cost:.2f}")
    print(f"   (This is a rough estimate - actual cost may vary)")

    # Confirm
    response = input("\n▶️  Run analysis? (y/n): ")
    if response.lower() != "y":
        print("\n❌ Cancelled")
        return 0

    # Run
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS")
    print("=" * 70)

    try:
        summarizer = FractalSummarizer(framework_config)
        result, metadata = summarizer.run(docs, analysis_config)

        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Layers: {metadata.total_layers}")
        print(f"   Total instances: {metadata.total_instances}")
        print(f"   Duration: {metadata.duration_seconds:.1f}s ({metadata.duration_seconds / 60:.1f} min)")
        print(f"   Cost: ${metadata.total_cost_usd:.2f}")
        print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,} tokens")

        print(f"\n📝 Final analysis preview (first 500 chars):")
        print("-" * 70)
        print(result[:500] + "...")
        print("-" * 70)

        output_dir = Path(analysis_config.output_dir)
        print(f"\n💾 Full output saved to:")
        print(f"   {output_dir / 'final_analysis.md'}")
        print(f"   {output_dir / 'run_metadata.json'}")

        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
