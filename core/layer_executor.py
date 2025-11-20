#!/usr/bin/env python3
"""
Layer execution for uniform-K random-batch summarization.

Executes one layer of the fractal process:
- Creates n = k·N/K instances
- Each instance samples K documents uniformly
- Calls LLM to compress
- Returns list of summaries (new documents for next layer)
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .batch_interface import BatchInterface, BatchRequest
from .document import Document


@dataclass
class LayerStats:
    """Statistics from executing a layer."""
    layer_num: int
    N: int  # Input documents
    K: int  # Documents per instance
    n_instances: int
    total_input_tokens: int
    total_output_tokens: int
    large_context_count: int
    compression_ratio: float

    def __str__(self):
        return (
            f"Layer {self.layer_num}: {self.N} docs → {self.n_instances} instances "
            f"(K={self.K}, {self.large_context_count} large context, "
            f"{self.compression_ratio:.2f}x compression)"
        )


def run_layer(
    documents: List[Document],
    k: float,
    K: int,
    r: float,
    T1: int,
    T2: int,
    layer_num: int,
    llm_caller: Callable[[str, str], Tuple[str, int]],
    prompt_builder: Callable[[List[Document], int, float, float], str],
    seed: int = None,
    batch_interface: BatchInterface = None,
    use_batch_api: bool = False
) -> Tuple[List[Document], LayerStats]:
    """
    Execute one layer of uniform-K random-batch summarization.

    Args:
        documents: Input documents for this layer
        k: Sampling density (expected reads per token)
        K: Documents per instance
        r: Target compression ratio
        T1: Small context budget (tokens)
        T2: Large context budget (tokens)
        layer_num: Current layer number
        llm_caller: Function(prompt, context_size) → (output_text, output_tokens)
        prompt_builder: Function(docs, layer, k, r) → prompt
        seed: Random seed for reproducibility
        batch_interface: BatchInterface for batch API (if use_batch_api=True)
        use_batch_api: If True, use batch API (50% cost, takes hours)

    Returns:
        Tuple of (output_documents, layer_stats)
    """
    if use_batch_api and batch_interface is None:
        raise ValueError("batch_interface required when use_batch_api=True")
    if seed is not None:
        np.random.seed(seed)

    N = len(documents)
    n_instances = int(k * N / K)

    if n_instances == 0:
        raise ValueError(f"n_instances = k·N/K = {k}·{N}/{K} = 0. Increase k or decrease K.")

    print(f"\nLayer {layer_num}: {N} documents, K={K}, {n_instances} instances")

    # Pre-sample all instances (same for both batch and real-time)
    instance_samples = []
    total_input_tokens = 0
    large_context_count = 0

    for instance_idx in range(n_instances):
        # Uniform-K sampling: sample K documents uniformly without replacement
        indices = np.random.choice(N, size=K, replace=False)
        selected_docs = [documents[i] for i in indices]

        # Check total size
        total_tokens = sum(doc.token_count for doc in selected_docs)
        total_input_tokens += total_tokens

        # Determine context size
        if total_tokens <= T1:
            context_size = "small"
        elif total_tokens <= T2:
            context_size = "large"
            large_context_count += 1
        else:
            doc_sizes = [doc.token_count for doc in selected_docs]
            raise ValueError(
                f"Instance {instance_idx} overflow: {total_tokens} > T2={T2}. "
                f"Document sizes: {doc_sizes}. K calibration failed."
            )

        instance_samples.append({
            'index': instance_idx,
            'selected_docs': selected_docs,
            'total_tokens': total_tokens,
            'context_size': context_size
        })

    # Process instances (batch or real-time)
    outputs = []
    total_output_tokens = 0

    if use_batch_api:
        # Batch API mode
        print(f"  Submitting {n_instances} instances to batch API...")

        # Create batch requests
        batch_requests = []
        for sample in instance_samples:
            prompt = prompt_builder(sample['selected_docs'], layer_num, k, r)
            batch_requests.append(
                BatchRequest(
                    custom_id=f"layer{layer_num}_instance{sample['index']:04d}",
                    prompt=prompt,
                    max_tokens=50_000,
                    context_size=sample['context_size']
                )
            )

        # Submit and wait (poll every 60 seconds)
        results = batch_interface.submit_and_wait(batch_requests, poll_interval=60)

        # Process results
        for custom_id, output_text, output_tokens in results:
            # Find matching sample
            instance_idx = int(custom_id.split('_instance')[-1])
            sample = instance_samples[instance_idx]

            total_output_tokens += output_tokens

            output_doc = Document(
                content=output_text,
                token_count=output_tokens,
                doc_id=custom_id,
                metadata={
                    "layer": layer_num,
                    "instance": instance_idx,
                    "input_docs": [doc.doc_id for doc in sample['selected_docs']],
                    "input_tokens": sample['total_tokens'],
                    "context_size": sample['context_size']
                }
            )

            outputs.append(output_doc)

        # Sort by instance index
        outputs.sort(key=lambda d: d.metadata['instance'])

    else:
        # Real-time mode
        for sample in instance_samples:
            instance_idx = sample['index']

            if (instance_idx + 1) % max(1, n_instances // 10) == 0 or instance_idx == 0:
                print(f"  Instance {instance_idx + 1}/{n_instances}...")

            # Build prompt
            prompt = prompt_builder(sample['selected_docs'], layer_num, k, r)

            # Call LLM
            output_text, output_tokens = llm_caller(prompt, sample['context_size'])
            total_output_tokens += output_tokens

            # Create output document
            output_doc = Document(
                content=output_text,
                token_count=output_tokens,
                doc_id=f"layer{layer_num}_instance{instance_idx:04d}",
                metadata={
                    "layer": layer_num,
                    "instance": instance_idx,
                    "input_docs": [doc.doc_id for doc in sample['selected_docs']],
                    "input_tokens": sample['total_tokens'],
                    "context_size": sample['context_size']
                }
            )

            outputs.append(output_doc)

    # Compute statistics
    compression_ratio = total_input_tokens / total_output_tokens if total_output_tokens > 0 else 0

    stats = LayerStats(
        layer_num=layer_num,
        N=N,
        K=K,
        n_instances=n_instances,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        large_context_count=large_context_count,
        compression_ratio=compression_ratio
    )

    print(f"  ✓ Complete: {len(outputs)} summaries")
    print(f"    Input: {total_input_tokens:,} tokens")
    print(f"    Output: {total_output_tokens:,} tokens")
    print(f"    Compression: {compression_ratio:.2f}x")
    print(f"    Large context: {large_context_count}/{n_instances} ({100*large_context_count/n_instances:.1f}%)")

    return outputs, stats


def _compression_to_words(r: float) -> str:
    """
    Convert compression ratio to natural language.

    Uses fractions for common values, percentages for others.

    Args:
        r: Compression ratio (e.g., 0.3 = compress to 30%)

    Returns:
        Natural language description (e.g., "approximately one-third")
    """
    # Map common ratios to fractions
    fraction_map = {
        0.20: "approximately one-fifth",
        0.25: "approximately one-quarter",
        0.30: "approximately one-third",
        0.33: "approximately one-third",
        0.40: "approximately two-fifths",
        0.50: "approximately half",
        0.60: "approximately three-fifths",
        0.67: "approximately two-thirds",
        0.75: "approximately three-quarters"
    }

    # Check for exact or near matches (within 0.01)
    for ratio, words in fraction_map.items():
        if abs(r - ratio) < 0.01:
            return words

    # Fall back to percentage for non-standard values
    return f"approximately {r:.0%}"


def build_default_layer_prompt(
    documents: List[Document],
    layer_num: int,
    k: float,
    r: float
) -> str:
    """
    Build default prompt for a layer instance.

    This is a simple template that can be overridden by users.

    Args:
        documents: Sampled documents for this instance
        layer_num: Current layer number
        k: Sampling density
        r: Target compression ratio

    Returns:
        Prompt string
    """
    # Concatenate documents
    doc_texts = []
    for doc in documents:
        doc_texts.append(f"## Document: {doc.doc_id}\n\n{doc.content}")

    combined = "\n\n---\n\n".join(doc_texts)

    # Convert compression ratio to natural language
    size_description = _compression_to_words(r)

    prompt = f"""You are analyzing a collection of documents through a multi-layer random-batch process.

LAYER {layer_num}:
You have been given {len(documents)} randomly sampled documents from the previous layer.
Each document in the corpus is read approximately {k} times per layer by different instances.

Your task:
- Create a comprehensive summary {size_description} the size of the input
- Preserve key themes, patterns, contradictions, and illustrative examples
- Note connections and tensions between documents
- Maintain information that helps recognize cross-cutting patterns

Documents:

{combined}

Provide your summary:
"""

    return prompt


if __name__ == "__main__":
    # Test with mock LLM
    print("="*70)
    print("LAYER EXECUTION TEST")
    print("="*70)

    # Create test documents
    from document import create_document_with_tokens

    test_docs = [
        create_document_with_tokens(
            content=f"This is test document {i}. " * 50,
            doc_id=f"test_{i:03d}"
        )
        for i in range(20)
    ]

    print(f"\nCreated {len(test_docs)} test documents")
    print(f"  Token counts: {[doc.token_count for doc in test_docs[:5]]}...")

    # Mock LLM caller
    def mock_llm(prompt: str, context_size: str) -> Tuple[str, int]:
        # Simulate compression to ~30% of input
        input_lines = len(prompt.split('\n'))
        output_lines = int(input_lines * 0.3)
        output = f"Mock summary of {output_lines} lines from {context_size} context."
        tokens = len(output.split())  # Rough approximation
        return output, tokens

    # Run one layer
    outputs, stats = run_layer(
        documents=test_docs,
        k=1.5,
        K=5,
        r=0.3,
        T1=100_000,
        T2=500_000,
        layer_num=1,
        llm_caller=mock_llm,
        prompt_builder=build_default_layer_prompt,
        seed=42
    )

    print(f"\n{stats}")
    print(f"\nOutput documents: {len(outputs)}")
    print(f"  First output ID: {outputs[0].doc_id}")
    print(f"  First output metadata: {outputs[0].metadata}")
