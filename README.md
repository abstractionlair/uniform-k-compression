# Uniform-K Compression

## The Problem

You have a large document collection—millions to billions of tokens—that won't fit in any LLM context window. You want to analyze it systematically, not just sample a few documents and hope they're representative. The binding constraints: context window size (finite) and cost (linear in tokens processed).

This framework solves that problem through multi-layer random sampling with two key properties: **per-token fairness** (every token has equal probability of being read) and **cross-document learning** (the model sees documents in combination, not isolation).

## How It Works

The core idea: layer a DAG of LLM calls, where each layer compresses documents through uniform random sampling.

**Layer structure**:
```
Documents (N) → Layer 1 (n₁ summaries) → Layer 2 (n₂ summaries) → ... → Final Analysis
```

At each layer:
1. Sample K documents uniformly at random (without replacement)
2. Send them to an LLM instance for compression
3. Repeat n times (where n = k·N/K) to ensure each document is read k times on average
4. Use the summaries as input to the next layer

**Why this works**: Uniform sampling ensures per-token fairness. Multiple instances per layer ensure cross-document connections emerge (documents A and B will appear together in some instances, A and C in others, etc.). Compression ratio r < 1 means logarithmic convergence: after log₁/ᵣ(N) layers, you're down to a manageable final size.

**The mathematics**:
- k = sampling density (each doc read k times per layer)
- r = compression ratio (each instance outputs r × input size)
- K = documents per instance (calibrated to fit in context window)
- n = k·N/K instances per layer
- α = k·r = effective layer compression (typically ~0.45)

Number of layers: log₁/α(N/Target) ≈ 3-5 for typical corpora.

## Key Design Decisions

**1. Uniform sampling over stratified/importance sampling**

Tradeoff: Uniform sampling wastes some probability mass on less-important documents but guarantees fairness and avoids bias from pre-filtering. Since we're reading each document k ≈ 1.5 times anyway, the waste is acceptable. More importantly, we don't know a priori which documents matter—cross-document patterns might be the most valuable signal.

**2. Adaptive K via bootstrap calibration**

The context window T constrains K·(average doc size). But average doc size changes each layer as we compress. Solution: recompute K after each layer by bootstrapping samples and measuring actual token usage. This is more robust than fixing K based on initial estimates.

**3. Separation of core framework and preprocessing**

Orthogonal concerns: the core algorithm operates on `List[Document]` (already loaded, already tokenized). How you get those documents (JSON conversion, chunking, etc.) is separate. This keeps the framework clean and makes it reusable across different document sources.

**4. Actual tokenization, not estimation**

Use tiktoken to count tokens exactly, not heuristics like "1 token ≈ 4 characters." The bootstrap calibration requires accurate token counts to avoid context window spills.

## Quick Start

### Installation

```bash
pip install -e .
```

This installs the package with dependencies: `anthropic`, `tiktoken`, `numpy`.

### Run an Example

**With Anthropic API** (costs ~$0.50-1.00):
```bash
export ANTHROPIC_API_KEY="your-key"
python examples/run_simple_example.py
```

**With Ollama** (free, local):
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:3b
python examples/run_ollama_example.py
```

Both examples analyze 5 Sherlock Holmes stories and demonstrate the full pipeline.

### Basic Usage

```python
from core import FrameworkConfig, AnalysisConfig, FractalSummarizer
from utilities import load_documents

# Load documents
documents = load_documents(
    directory_path="path/to/text/files",
    pattern="*.txt"
)

# Configure
framework_config = FrameworkConfig(
    k=1.5,           # Sampling density
    r=0.3,           # Compression ratio
    T1=154_000,      # Small context budget
    T2=769_000,      # Large context budget
)

analysis_config = AnalysisConfig(
    name="My Analysis",
    layer_prompt_template="Analyze {num_docs} documents...\n{documents}",
    final_synthesis_prompt="Synthesize findings:",
    output_dir="output/my_analysis"
)

# Run
summarizer = FractalSummarizer(framework_config)
result, metadata = summarizer.run(documents, analysis_config)

print(f"Layers: {metadata.total_layers}, Cost: ${metadata.total_cost_usd:.2f}")
```

### Iterative Refinement with Commentary

The framework supports iterative improvement through user feedback. After reviewing an analysis, you can provide commentary that guides subsequent runs:

```python
from core import CommentaryManager

# 1. Run initial analysis
analysis_config = AnalysisConfig(
    name="Initial Run",
    layer_prompt_template="...",
    final_synthesis_prompt="...",
    output_dir="output/run1"
)

result1, metadata1 = summarizer.run(documents, analysis_config)

# 2. Review output and create commentary file
# Generate template:
# python -m core.commentary_manager template my_feedback.md
#
# Fill in template with feedback:
# - What was accurate
# - What was missing
# - What was misunderstood
# - Important context

# 3. Re-run with commentary
analysis_config_v2 = AnalysisConfig(
    name="Second Run",
    layer_prompt_template="...",
    final_synthesis_prompt="...",
    output_dir="output/run2",
    commentary_file="my_feedback.md"  # ← Include commentary
)

result2, metadata2 = summarizer.run(documents, analysis_config_v2)
# Results improve with each iteration
```

**How it works**: Commentary is incorporated into prompts at all layers (Layer 1, Layer 2+, Final Synthesis). The LLM sees your feedback and adjusts its analysis accordingly. Each iteration builds on previous insights.

**Example workflow**:
```bash
# Run the iterative refinement example
python examples/run_iterative_example.py
```

This demonstrates a complete two-run workflow with commentary, showing how feedback improves analysis quality.

## Project Structure

```
uniform-k-compression/
├── core/                        # Core algorithm
│   ├── document.py              # Document data model (immutable)
│   ├── k_calibrator.py          # Bootstrap K selection
│   ├── layer_executor.py        # Layer execution: sample → LLM → collect
│   ├── llm_interface.py         # Anthropic API wrapper
│   ├── ollama_interface.py      # Ollama local LLM wrapper
│   ├── batch_interface.py       # Batch API support (50% cost savings)
│   ├── commentary_manager.py    # Iterative refinement with user feedback
│   ├── fractal_summarizer.py    # Main orchestration
│   └── config.py                # Configuration classes
│
├── utilities/                   # Preprocessing tools (optional)
│   ├── document_loader.py       # Directory → List[Document]
│   ├── cooperative_chunker.py   # LLM-based document splitting
│   ├── json_converter.py        # Claude JSON exports → text
│   └── chatgpt_cleaner.py       # Clean ChatGPT tool call output
│
├── tests/                       # Test suite
│   ├── test_document.py         # Document model tests
│   ├── test_config.py           # Configuration tests
│   ├── test_k_calibrator.py     # K calibration tests
│   ├── test_loader.py           # Document loading tests
│   ├── test_layer_executor_mock.py  # Layer execution (mocked)
│   ├── test_llm_interface_mock.py   # LLM interface (mocked)
│   ├── test_end_to_end_ollama.py    # Full pipeline (real Ollama)
│   └── test_ollama_question.py      # Question-answering demo
│
├── examples/                    # Working examples
│   ├── run_simple_example.py    # Anthropic API example
│   ├── run_ollama_example.py    # Ollama local example
│   ├── run_iterative_example.py # Iterative refinement demo
│   └── sample_data/             # Public domain test data
│
└── configs/                     # Example configurations
    ├── framework_default.json
    ├── analysis_template.json
    ├── example_meeting_notes.json
    └── example_research_papers.json
```

## Configuration

### Framework Config (Algorithm Parameters)

```json
{
  "k": 1.5,
  "r": 0.3,
  "T1": 154000,
  "T2": 769000,
  "target_convergence": 700000,
  "model": "claude-sonnet-4-5-20250929",
  "use_batch_api": false
}
```

- **k**: Sampling density (how many times each document is read per layer)
- **r**: Compression ratio (output = r × input size)
- **T1/T2**: Context budgets (small/large windows)
- **target_convergence**: Stop when corpus shrinks below this
- **use_batch_api**: Trade latency for 50% cost savings

### Analysis Config (Task-Specific)

```json
{
  "name": "Meeting Notes Analysis",
  "description": "Extract key decisions and action items",
  "layer_prompt_template": "Analyze {num_docs} documents...\n{documents}",
  "final_synthesis_prompt": "Synthesize findings:",
  "output_dir": "output/meeting_notes",
  "commentary_file": null
}
```

**Parameters**:
- **name**: Analysis name for tracking
- **layer_prompt_template**: Prompt template for each layer (variables: `{documents}`, `{layer_num}`, `{k}`, `{r}`, `{num_docs}`)
- **final_synthesis_prompt**: Prompt for final synthesis
- **output_dir**: Directory for outputs
- **commentary_file** (optional): Path to markdown file with user feedback from previous run

See `configs/example_*.json` for complete examples.

## Testing and Development

### Run Tests

```bash
# All tests (no API key needed)
pytest tests/ -v

# Specific test
pytest tests/test_document.py -v

# With Ollama (if installed)
pytest tests/test_end_to_end_ollama.py -v
```

The test suite includes:
- **Unit tests**: Document model, config, K calibration (no API needed)
- **Mock tests**: Layer execution, LLM interface (no API needed)
- **Ollama tests**: Full pipeline with local LLM (free, skips if Ollama unavailable)

### Local Development with Ollama

Ollama lets you test the full pipeline for free using local models:

```bash
# Install Ollama
# Download from https://ollama.ai

# Pull a model
ollama pull qwen2.5:3b

# Run example
python examples/run_ollama_example.py
```

See [docs/OLLAMA_TESTING.md](docs/OLLAMA_TESTING.md) for details.

## Cost and Performance

For k=1.5, r=0.3 (default parameters):

**Example: 5,000 documents, 20M tokens**
- Layer 1: 750 instances → 9M tokens output
- Layer 2: 113 instances → 4M tokens output
- Layer 3: 42 instances → 1.8M tokens output
- Layer 4: 20 instances → 800K tokens (converged)

**Total**: ~925 instances

| Method | Cost | Time | Notes |
|--------|------|------|-------|
| Anthropic Real-time | $60-120 | 6-8 hours | Full quality |
| Anthropic Batch | $30-60 | 2-4 days | 50% discount |
| Ollama (local) | $0 | 2-4 hours | Good for testing |

Scaling: Cost scales linearly with k, logarithmically with corpus size.

## Implementation Details

### What "Uniform-K" Means

**CORRECT** (this implementation):
- Sample K **whole documents** uniformly
- Read entire documents, not fragments
- Create n = k·N/K instances
- Each document appears in k/K ≈ 1.5 instances per layer

**INCORRECT** interpretation:
- Sample k lines from each document
- This would be *stratified* sampling, not uniform
- Breaks cross-document learning
- Violates per-token fairness

### Why K Is Adaptive

Context budget T constrains K through: K·E[doc_size] ≤ T

But E[doc_size] changes each layer (compression). Can't just fix K at the start.

Solution: After each layer, bootstrap sample thousands of random K-tuples, measure actual token usage, find the K where 95% of samples fit in budget.

Tradeoff: K too small → fewer cross-document connections. K too large → context spills (wasted API calls). Bootstrap targets 5% spill rate.

### When to Chunk Documents

If you have documents > 50K tokens, K might drop below 10 (too few cross-document connections).

Solution: Use `cooperative_chunker.py` to split documents before analysis. The chunker uses an LLM to find natural break points and adds editor's notes for continuity.

### Batch API vs Real-Time

**Batch API**: 50% cost reduction, but hours of latency per layer.
**Real-time API**: Full cost, but minutes per layer.

Use batch for production runs where you can wait. Use real-time for development/iteration.

## Limitations and Tradeoffs

**What this optimizes for**: Large corpora (>1M tokens) where you want systematic coverage, not ad-hoc sampling.

**Tradeoffs**:
- Cost: k=1.5 means reading each document 1.5× per layer. More expensive than naive "read N random documents once."
- Latency: Multi-layer means sequential LLM calls. Can't parallelize across layers.
- Compression quality: r=0.3 means aggressive compression. Some nuance will be lost. Prompts matter.

**When not to use this**:
- Small corpora (<100 documents): Just read them all directly.
- When you know exactly which documents matter: Use targeted sampling instead.
- When you need guaranteed lossless preservation: This is lossy compression.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and testing guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
