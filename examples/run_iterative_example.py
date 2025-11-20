#!/usr/bin/env python3
"""
Iterative refinement example: Run analysis, get feedback, re-run with commentary.

This demonstrates the iterative workflow:
1. Run initial analysis
2. Review output and create commentary file with feedback
3. Re-run analysis with commentary incorporated into prompts
4. Results improve with each iteration

For this example, we'll use Ollama (free, local) to keep it accessible.
If you prefer Anthropic API, just remove the OllamaFractalSummarizer class
and use the standard FractalSummarizer.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    FrameworkConfig,
    AnalysisConfig,
    FractalSummarizer,
    OllamaInterface,
    is_ollama_available,
    CommentaryManager,
)
from utilities import load_documents


class OllamaFractalSummarizer(FractalSummarizer):
    """FractalSummarizer using Ollama for local, free execution."""

    def __init__(self, config: FrameworkConfig, model: str = "qwen3:8b"):
        self.config = config
        self.llm = OllamaInterface(model=model)
        self.batch = None
        from core.document import Tokenizer
        self.tokenizer = Tokenizer()


def main():
    """Run iterative analysis example."""

    # Check Ollama availability
    if not is_ollama_available():
        print("❌ Ollama not available. Please install from https://ollama.ai")
        print("   Then run: ollama pull qwen3:8b")
        return

    # Load sample documents (using Sherlock Holmes stories)
    sample_data_dir = Path(__file__).parent / "sample_data"

    print("="*70)
    print("ITERATIVE REFINEMENT EXAMPLE")
    print("="*70)
    print("\nThis example demonstrates iterative improvement through commentary.")
    print("We'll analyze 3 Sherlock Holmes stories twice:")
    print("  1. Initial run (no commentary)")
    print("  2. Second run (with commentary from example file)")
    print()

    # Load 3 stories for quick demo
    documents = load_documents(
        directory_path=str(sample_data_dir),
        pattern="*.txt",
        limit=3  # Just 3 stories for quick demo
    )

    print(f"Loaded {len(documents)} stories ({sum(d.token_count for d in documents):,} tokens)\n")

    # Framework config (small settings for quick demo)
    framework_config = FrameworkConfig(
        k=1.5,
        r=0.4,
        T1=20_000,
        T2=80_000,
        target_convergence=5_000,
    )

    # =========================================================================
    # RUN 1: Initial analysis (no commentary)
    # =========================================================================

    print("="*70)
    print("RUN 1: Initial Analysis (No Commentary)")
    print("="*70)
    print()

    analysis_config_run1 = AnalysisConfig(
        name="Holmes Stories - Run 1",
        description="Initial analysis without commentary",
        layer_prompt_template="""Analyze {num_docs} Sherlock Holmes stories.

Extract key themes, character traits, and notable plot elements.
Compress to ~{r:.0%} while preserving important details.

Stories:
{documents}

Summary:""",
        final_synthesis_prompt="""Synthesize the analysis of these Sherlock Holmes stories.

What are the key themes, character traits, and patterns across the stories?

Summaries to synthesize:""",
        output_dir="output/iterative_demo/run1"
    )

    summarizer = OllamaFractalSummarizer(framework_config, model="qwen3:8b")
    result1, metadata1 = summarizer.run(documents, analysis_config_run1)

    print("\n" + "="*70)
    print("RUN 1 COMPLETE")
    print("="*70)
    print(f"Output saved to: output/iterative_demo/run1/final_analysis.md")
    print(f"Duration: {metadata1.duration_seconds/60:.1f} minutes")
    print()
    print("→ Review the output and create a commentary file with feedback.")
    print("  For this demo, we'll use a pre-made example commentary.")
    print()

    # =========================================================================
    # Create example commentary
    # =========================================================================

    commentary_dir = Path("output/iterative_demo")
    commentary_dir.mkdir(parents=True, exist_ok=True)
    commentary_file = commentary_dir / "example_commentary.md"

    example_commentary = """# Commentary on Run 1

**Run being commented on:** Holmes Stories - Run 1

## What Was Accurate

The analysis correctly identified Holmes' deductive reasoning as central to the stories.
The focus on observation and logical inference was spot-on.

## What Was Missing

The analysis didn't capture the emotional dynamics between Holmes and Watson. Their
friendship and Watson's role as narrator are crucial to the stories' appeal.

Also missing: the Victorian London setting and how it shapes the mysteries.

## What Was Misunderstood

The analysis suggested Holmes is purely rational, but he actually has moments of
passion and even theatricality. He's more complex than "cold logic machine."

## Important Context

These stories are detective fiction, but they're also character studies and
period pieces. The genre conventions matter - readers expect certain beats
and revelations.

## Specific Corrections

- Holmes isn't emotionless - he gets excited, frustrated, and even appreciative
- Watson isn't just a sidekick - he's the emotional heart and audience surrogate
- The mysteries aren't just puzzles - they reveal Victorian social anxieties
"""

    with open(commentary_file, 'w') as f:
        f.write(example_commentary)

    print(f"Created example commentary: {commentary_file}")
    print()

    # =========================================================================
    # RUN 2: Re-run with commentary
    # =========================================================================

    print("="*70)
    print("RUN 2: Analysis with Commentary")
    print("="*70)
    print()

    analysis_config_run2 = AnalysisConfig(
        name="Holmes Stories - Run 2",
        description="Re-run with commentary from Run 1",
        layer_prompt_template="""Analyze {num_docs} Sherlock Holmes stories.

Extract key themes, character traits, and notable plot elements.
Compress to ~{r:.0%} while preserving important details.

Stories:
{documents}

Summary:""",
        final_synthesis_prompt="""Synthesize the analysis of these Sherlock Holmes stories.

What are the key themes, character traits, and patterns across the stories?

Summaries to synthesize:""",
        output_dir="output/iterative_demo/run2",
        commentary_file=str(commentary_file)  # ← Include commentary
    )

    # Reset LLM usage tracking for clean Run 2 stats
    summarizer = OllamaFractalSummarizer(framework_config, model="qwen3:8b")
    result2, metadata2 = summarizer.run(documents, analysis_config_run2)

    print("\n" + "="*70)
    print("RUN 2 COMPLETE")
    print("="*70)
    print(f"Output saved to: output/iterative_demo/run2/final_analysis.md")
    print(f"Duration: {metadata2.duration_seconds/60:.1f} minutes")
    print()

    # =========================================================================
    # Compare results
    # =========================================================================

    print("="*70)
    print("COMPARISON")
    print("="*70)
    print("\nRun 1 (no commentary):")
    print(f"  Layers: {metadata1.total_layers}")
    print(f"  Duration: {metadata1.duration_seconds/60:.1f} min")
    print(f"  Output length: {len(result1)} chars")
    print()
    print("Run 2 (with commentary):")
    print(f"  Layers: {metadata2.total_layers}")
    print(f"  Duration: {metadata2.duration_seconds/60:.1f} min")
    print(f"  Output length: {len(result2)} chars")
    print()
    print("→ Compare the outputs to see how commentary influenced the analysis.")
    print(f"   Run 1: output/iterative_demo/run1/final_analysis.md")
    print(f"   Run 2: output/iterative_demo/run2/final_analysis.md")
    print()
    print("Key takeaway: Run 2 should better address the emotional dynamics,")
    print("Victorian context, and character complexity mentioned in commentary.")
    print()

    # =========================================================================
    # Show how to create commentary template
    # =========================================================================

    print("="*70)
    print("CREATING YOUR OWN COMMENTARY")
    print("="*70)
    print("\nTo create commentary for your own analysis:")
    print()
    print("1. Generate a template:")
    print("   python -m core.commentary_manager template my_commentary.md")
    print()
    print("2. Fill in the template with your feedback:")
    print("   - What was accurate")
    print("   - What was missing")
    print("   - What was misunderstood")
    print("   - Important context")
    print()
    print("3. Re-run with commentary_file parameter:")
    print("   analysis_config = AnalysisConfig(")
    print("       ...,")
    print("       commentary_file='my_commentary.md'")
    print("   )")
    print()
    print("4. Iterate! Each run can incorporate previous commentary.")
    print()


if __name__ == "__main__":
    main()
