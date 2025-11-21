#!/usr/bin/env python3
"""
Main orchestration for uniform-K fractal summarization.

Coordinates the multi-layer process with adaptive K selection.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .commentary_manager import CommentaryManager
from .config import AnalysisConfig, FrameworkConfig
from .document import Document, Tokenizer
from .k_calibrator import find_optimal_K
from .layer_executor import LayerStats, _compression_to_words, run_layer
from .provider_factory import create_provider


@dataclass
class RunMetadata:
    """Metadata from a complete fractal summarization run."""
    analysis_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_layers: int
    total_instances: int
    initial_documents: int
    initial_tokens: int
    final_documents: int
    final_tokens: int
    total_cost_usd: float
    layer_stats: List[dict]  # LayerStats converted to dicts

    def save(self, path: Path):
        """Save metadata to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class FractalSummarizer:
    """
    Main class for uniform-K fractal summarization.

    Coordinates the multi-layer process with adaptive K selection.
    """

    def __init__(self, config: FrameworkConfig):
        """
        Initialize summarizer with framework configuration.

        Args:
            config: FrameworkConfig with k, r, T1, T2, provider, etc.
        """
        self.config = config

        # Create provider instance
        self.provider = create_provider(
            provider_name=config.provider,
            model=config.model,
            large_context_model=config.large_context_model,
            temperature=config.temperature
        )

        # Legacy attributes for backwards compatibility
        self.llm = self.provider
        self.batch = self.provider if config.use_batch_api and self.provider.supports_batch() else None

        self.tokenizer = Tokenizer()

    def run(
        self,
        documents: List[Document],
        analysis_config: AnalysisConfig,
        prompt_builder: Optional[Callable] = None
    ) -> Tuple[str, RunMetadata]:
        """
        Run complete fractal summarization process.

        Args:
            documents: Input document collection (already tokenized)
            analysis_config: Analysis configuration with prompts, output dir
            prompt_builder: Optional custom prompt builder function

        Returns:
            Tuple of (final_analysis, run_metadata)
        """
        if prompt_builder is None:
            prompt_builder = self._build_layer_prompt_from_template

        start_time = datetime.now()

        # Initialize
        output_dir = Path(analysis_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load commentary if provided
        commentary = None
        if analysis_config.commentary_file:
            commentary = CommentaryManager(analysis_config.commentary_file)
            print(f"\n📝 Using commentary from: {analysis_config.commentary_file}")

        initial_doc_count = len(documents)
        initial_tokens = sum(doc.token_count for doc in documents)

        print("="*70)
        print(f"FRACTAL SUMMARIZATION: {analysis_config.name}")
        print("="*70)
        print("\nFramework config:")
        print(f"  k={self.config.k}, r={self.config.r}, α≈{self.config.alpha:.2f}")
        print(f"  T1={self.config.T1:,}, T2={self.config.T2:,}")
        print("\nInitial corpus:")
        print(f"  {initial_doc_count} documents")
        print(f"  {initial_tokens:,} tokens (~{initial_tokens/1e6:.1f}M)")

        # Track layers
        current_docs = documents
        layer_num = 1
        all_layer_stats = []

        # Multi-layer loop
        while True:
            total_tokens = sum(doc.token_count for doc in current_docs)

            print(f"\n{'='*70}")
            print(f"LAYER {layer_num} INPUT")
            print(f"{'='*70}")
            print(f"  Documents: {len(current_docs)}")
            print(f"  Total tokens: {total_tokens:,} (~{total_tokens/1e6:.1f}M)")

            # Check convergence
            if total_tokens < self.config.target_convergence:
                print(f"\n✓ CONVERGED! {total_tokens:,} tokens < target {self.config.target_convergence:,}")
                break

            # Compute optimal K for this layer
            doc_lengths = [doc.token_count for doc in current_docs]
            K = find_optimal_K(
                doc_token_counts=doc_lengths,
                T1=self.config.T1,
                T2=self.config.T2,
                target_spill_rate=self.config.target_spill_rate,
                n_bootstrap=self.config.bootstrap_iterations,
                verbose=True
            )

            # Save layer input (for debugging/inspection)
            layer_input_file = output_dir / f"layer{layer_num}_input.json"
            self._save_layer_input(current_docs, layer_input_file)

            # Run layer
            current_docs, stats = run_layer(
                documents=current_docs,
                k=self.config.k,
                K=K,
                r=self.config.r,
                T1=self.config.T1,
                T2=self.config.T2,
                layer_num=layer_num,
                llm_caller=self._llm_call_wrapper,
                prompt_builder=lambda docs, ln, k, r: prompt_builder(
                    docs, ln, k, r, analysis_config.layer_prompt_template, commentary
                ),
                seed=42,  # For reproducibility
                batch_interface=self.batch,
                use_batch_api=self.config.use_batch_api
            )

            all_layer_stats.append(asdict(stats))

            # Save layer output
            layer_output_file = output_dir / f"layer{layer_num}_output.json"
            self._save_layer_output(current_docs, stats, layer_output_file)

            layer_num += 1

            # Safety check
            if layer_num > 20:
                print("\n⚠️  Warning: Exceeded 20 layers, stopping")
                break

        # Final synthesis
        print(f"\n{'='*70}")
        print("FINAL SYNTHESIS")
        print(f"{'='*70}")
        print(f"  Combining {len(current_docs)} final summaries...")

        final_analysis = self._run_final_synthesis(current_docs, analysis_config, commentary)

        # Save final output
        final_output_file = output_dir / "final_analysis.md"
        with open(final_output_file, 'w') as f:
            f.write(final_analysis)

        print(f"\n✓ Saved final analysis to: {final_output_file}")

        # Create metadata
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        total_instances = sum(stats_dict['n_instances'] for stats_dict in all_layer_stats)

        metadata = RunMetadata(
            analysis_name=analysis_config.name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_layers=layer_num - 1,
            total_instances=total_instances,
            initial_documents=initial_doc_count,
            initial_tokens=initial_tokens,
            final_documents=len(current_docs),
            final_tokens=sum(doc.token_count for doc in current_docs),
            total_cost_usd=self.provider.calculate_cost(
                self.provider.get_total_usage(),
                self.config.model
            ),
            layer_stats=all_layer_stats
        )

        metadata.save(output_dir / "run_metadata.json")

        # Print summary
        print(f"\n{'='*70}")
        print("RUN COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration: {duration/60:.1f} minutes")
        print(f"  Layers: {metadata.total_layers}")
        print(f"  Total instances: {total_instances}")
        print(f"  Compression: {initial_tokens:,} → {metadata.final_tokens:,} ({initial_tokens/metadata.final_tokens:.1f}x)")
        print(f"  Total cost: ${metadata.total_cost_usd:.2f}")

        return final_analysis, metadata

    def _llm_call_wrapper(self, prompt: str, context_size: str) -> Tuple[str, int]:
        """
        Wrapper for LLM calls that matches layer_executor signature.

        Args:
            prompt: Input prompt
            context_size: "small" or "large"

        Returns:
            Tuple of (output_text, output_token_count)
        """
        output_text, output_tokens, usage = self.llm.call(prompt, context_size)
        return output_text, output_tokens

    def _build_layer_prompt_from_template(
        self,
        documents: List[Document],
        layer_num: int,
        k: float,
        r: float,
        template: str,
        commentary: Optional[CommentaryManager] = None
    ) -> str:
        """
        Build layer prompt from analysis config template.

        Args:
            documents: Sampled documents
            layer_num: Current layer
            k: Sampling density
            r: Compression ratio
            template: Template string from analysis config
            commentary: Optional commentary manager for iterative refinement

        Returns:
            Complete prompt
        """
        # Concatenate documents
        doc_texts = []
        for doc in documents:
            doc_texts.append(f"## Document: {doc.doc_id}\n\n{doc.content}")

        combined = "\n\n---\n\n".join(doc_texts)

        # Convert compression ratio to natural language
        r_words = _compression_to_words(r)

        # Format template
        # Template can use: {documents}, {layer_num}, {k}, {r}, {r_words}, {num_docs}
        prompt = template.format(
            documents=combined,
            layer_num=layer_num,
            k=k,
            r=r,
            r_words=r_words,
            num_docs=len(documents)
        )

        # Augment with commentary if provided
        if commentary:
            layer_type = "layer1" if layer_num == 1 else "layer2+"
            prompt = commentary.augment_prompt(prompt, layer_type)

        return prompt

    def _run_final_synthesis(
        self,
        final_layer_docs: List[Document],
        analysis_config: AnalysisConfig,
        commentary: Optional[CommentaryManager] = None
    ) -> str:
        """
        Run final synthesis on converged documents.

        Note: This always uses direct API (not batch) because it's a single
        high-value call where immediate results are preferred. Batch API would
        add significant latency for minimal cost savings on one request.

        Args:
            final_layer_docs: Documents from final layer
            analysis_config: Analysis configuration
            commentary: Optional commentary manager for iterative refinement

        Returns:
            Final analysis text
        """
        # Combine all final documents
        combined = "\n\n".join([
            f"## Summary {i+1}/{len(final_layer_docs)}: {doc.doc_id}\n\n{doc.content}"
            for i, doc in enumerate(final_layer_docs)
        ])

        # Build final prompt
        prompt = f"""{analysis_config.final_synthesis_prompt}

---

Here are the {len(final_layer_docs)} final summaries from the last layer:

{combined}
"""

        # Augment with commentary if provided
        if commentary:
            prompt = commentary.augment_prompt(prompt, "final")

        # Call with small context for final synthesis (usually small enough)
        # If final docs exceed T1, this could be changed to "large"
        output, _, usage = self.llm.call(prompt, "small", max_tokens=64_000)

        print(f"  Final synthesis: {usage.output_tokens:,} tokens")
        cost = self.provider.calculate_cost(usage, self.config.model)
        print(f"  Cost: ${cost:.2f}")

        return output

    def _save_layer_input(self, documents: List[Document], path: Path):
        """Save layer input documents for inspection."""
        data = {
            "count": len(documents),
            "total_tokens": sum(doc.token_count for doc in documents),
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "token_count": doc.token_count,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                }
                for doc in documents
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_layer_output(self, documents: List[Document], stats: LayerStats, path: Path):
        """Save layer output documents and stats."""
        data = {
            "stats": asdict(stats),
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "token_count": doc.token_count,
                    "metadata": doc.metadata,
                    "content": doc.content
                }
                for doc in documents
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Example usage (requires documents and API key)
    from .config import AnalysisConfig, FrameworkConfig
    from .document import create_document_with_tokens

    # Create test documents
    print("Creating test documents...")
    tokenizer = Tokenizer()
    test_docs = [
        create_document_with_tokens(
            content=f"This is test document {i}. " * 100,
            doc_id=f"test_{i:03d}",
            tokenizer=tokenizer
        )
        for i in range(10)
    ]

    print(f"  {len(test_docs)} documents, {sum(d.token_count for d in test_docs):,} tokens")

    # Create configs
    framework_config = FrameworkConfig(k=1.5, r=0.3)

    analysis_config = AnalysisConfig(
        name="Test Run",
        layer_prompt_template="""Analyze these {num_docs} documents at layer {layer_num}.
Compress to ~{r:.0%} of input size while preserving key themes.

{documents}

Provide summary:""",
        final_synthesis_prompt="Synthesize all findings:",
        output_dir="output/test_run"
    )

    # Run (will fail if no API key, but demonstrates interface)
    try:
        summarizer = FractalSummarizer(framework_config)
        result, metadata = summarizer.run(test_docs, analysis_config)

        print(f"\nFinal analysis length: {len(result)} chars")
        print(f"Metadata: {metadata.total_layers} layers, ${metadata.total_cost_usd:.2f}")

    except ValueError as e:
        print(f"\nSkipping actual run (likely no API key): {e}")
        print("Framework is ready - just needs API key to execute.")
