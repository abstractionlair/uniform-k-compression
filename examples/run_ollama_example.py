#!/usr/bin/env python3
"""
Example using Ollama for local testing (no API costs).

This demonstrates the same fractal summarization pipeline but using
a local Ollama model instead of the Anthropic API.

Requirements:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull qwen2.5:3b

Usage:
    python examples/run_ollama_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import (
    FrameworkConfig,
    AnalysisConfig,
    FractalSummarizer,
    OllamaInterface,
    is_ollama_available,
)
from utilities import load_documents


class OllamaFractalSummarizer(FractalSummarizer):
    """FractalSummarizer that uses Ollama instead of Anthropic."""

    def __init__(self, config: FrameworkConfig, model: str = "qwen2.5:3b"):
        # Don't call super().__init__ to avoid Anthropic API key requirement
        self.config = config
        self.llm = OllamaInterface(model=model)
        self.batch = None  # No batch support for Ollama
        from core.document import Tokenizer
        self.tokenizer = Tokenizer()


def main():
    print("=" * 70)
    print("FRACTAL SUMMARIZATION - OLLAMA EXAMPLE (FREE!)")
    print("=" * 70)

    # Check Ollama availability
    if not is_ollama_available():
        print("\n❌ Error: Ollama not running")
        print("\n📥 Install Ollama:")
        print("   1. Download from: https://ollama.ai")
        print("   2. Install and start the app")
        print("   3. Pull a model: ollama pull qwen2.5:3b")
        print("\nThen run this script again.")
        return 1

    print("\n✅ Ollama is running")

    # Load sample documents
    print("\n📚 Loading 5 Sherlock Holmes stories...")
    sample_data_dir = Path(__file__).parent / "sample_data"

    docs = load_documents(
        directory_path=str(sample_data_dir),
        pattern="*.txt"
    )

    print(f"   Loaded {len(docs)} documents")
    print(f"   Total tokens: {sum(d.token_count for d in docs):,}")

    # Configure framework (smaller contexts for local models)
    framework_config = FrameworkConfig(
        k=1.5,
        r=0.4,  # Less aggressive compression
        T1=8_000,  # Smaller context windows
        T2=32_000,
        target_convergence=20_000,
        bootstrap_iterations=500,
    )

    # Configure analysis
    analysis_config = AnalysisConfig(
        name="Sherlock Holmes Stories (Ollama)",
        description="Example using local Ollama model",
        layer_prompt_template="""Analyze {num_docs} Sherlock Holmes stories.

Compress to about {r:.0%} of original length while keeping:
- Main plot points
- Character names
- How mysteries were solved

Stories:
{documents}

Summary:""",
        final_synthesis_prompt="""Synthesize these summaries into a brief overview:
1. Common themes across stories
2. Holmes' investigative methods
3. Types of cases

Keep it concise (2-3 paragraphs).""",
        output_dir="examples/output_ollama",
    )

    print(f"\n⚙️  Framework configuration:")
    print(f"   Model: qwen2.5:3b (local)")
    print(f"   k={framework_config.k}, r={framework_config.r}")
    print(f"   Context: T1={framework_config.T1:,}, T2={framework_config.T2:,}")
    print(f"\n💰 Cost: $0.00 (running locally!)")

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
        summarizer = OllamaFractalSummarizer(framework_config, model="qwen2.5:3b")
        result, metadata = summarizer.run(docs, analysis_config)

        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Model: qwen2.5:3b (Ollama)")
        print(f"   Layers: {metadata.total_layers}")
        print(f"   Total instances: {metadata.total_instances}")
        print(f"   Duration: {metadata.duration_seconds:.1f}s ({metadata.duration_seconds / 60:.1f} min)")
        print(f"   Cost: $0.00 (free!)")
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

    except RuntimeError as e:
        print(f"\n❌ Ollama error: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Make sure Ollama is running")
        print("   - Pull the model: ollama pull qwen2.5:3b")
        print("   - Check: ollama list")
        return 1
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
