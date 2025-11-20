# Local Testing with Ollama

Run fractal summarization **completely free** using local Ollama models instead of the Anthropic API.

## Why Ollama?

- **$0 cost** - No API fees
- **Fast iteration** - No network latency
- **Privacy** - Documents stay on your machine
- **Testing** - Verify the pipeline before using paid APIs

## Quick Start

### 1. Install Ollama

Download and install from [ollama.ai](https://ollama.ai)

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download the app from https://ollama.ai
```

### 2. Pull a Model

We recommend `qwen2.5:3b` for good balance of speed and quality:

```bash
ollama pull qwen2.5:3b
```

**Other good options**:
- `qwen2.5:7b` - Better quality, slower
- `qwen2.5:1.5b` - Faster, lower quality
- `llama3.2:3b` - Alternative fast model

### 3. Verify Ollama is Running

```bash
ollama list
# Should show: qwen2.5:3b
```

### 4. Run the Example

```bash
python examples/run_ollama_example.py
```

This analyzes 5 Sherlock Holmes stories completely locally - no API key needed!

## Using Ollama in Your Code

### Basic Usage

```python
from core import OllamaInterface, FrameworkConfig, AnalysisConfig, FractalSummarizer
from utilities import load_documents

# Check if Ollama is available
from core import is_ollama_available
if not is_ollama_available():
    print("Install Ollama from https://ollama.ai")
    exit(1)

# Create Ollama interface
ollama = OllamaInterface(model="qwen2.5:3b")

# Test it
output, tokens, usage = ollama.call(
    "Summarize: The quick brown fox jumps over the lazy dog",
    context_size="small",
    max_tokens=50
)

print(output)
print(f"Cost: ${usage.cost_usd:.2f}")  # Always $0.00
```

### Full Pipeline with Ollama

```python
from core import FrameworkConfig, AnalysisConfig, FractalSummarizer, OllamaInterface
from core.document import Tokenizer

class OllamaFractalSummarizer(FractalSummarizer):
    """FractalSummarizer that uses Ollama instead of Anthropic."""

    def __init__(self, config: FrameworkConfig, model: str = "qwen2.5:3b"):
        self.config = config
        self.llm = OllamaInterface(model=model)
        self.batch = None
        self.tokenizer = Tokenizer()

# Use smaller context windows for local models
framework_config = FrameworkConfig(
    k=1.5,
    r=0.4,  # Less aggressive compression
    T1=8_000,  # Smaller contexts
    T2=32_000,
    target_convergence=20_000,
)

# Run as normal
summarizer = OllamaFractalSummarizer(framework_config)
result, metadata = summarizer.run(documents, analysis_config)
```

## Performance Comparison

### qwen2.5:3b (recommended)

- **Speed**: ~50-100 tokens/sec on M1/M2 Mac
- **Quality**: Good for testing, decent summaries
- **Context**: Handles 8K-32K tokens well
- **Use case**: Development, testing, small corpora

### Anthropic Claude Sonnet

- **Speed**: ~30-60 tokens/sec (network dependent)
- **Quality**: Excellent, research-grade summaries
- **Context**: Handles 200K-1M tokens
- **Use case**: Production, large corpora, final analysis

### Cost Comparison (5,000 documents, 20M tokens)

| Provider | Cost | Time | Notes |
|----------|------|------|-------|
| Ollama (qwen2.5:3b) | $0 | 2-4 hours | Local machine |
| Claude Sonnet | $60-120 | 6-8 hours | Real-time API |
| Claude Batch | $30-60 | 2-4 days | 50% discount |

## Configuration Tips for Ollama

### Adjust Context Windows

Local models typically handle smaller contexts:

```python
framework_config = FrameworkConfig(
    T1=8_000,   # Instead of 154_000
    T2=32_000,  # Instead of 769_000
    target_convergence=20_000,
)
```

### Less Aggressive Compression

Smaller models benefit from gentler compression:

```python
framework_config = FrameworkConfig(
    r=0.4,  # Instead of 0.3
)
```

### Faster K Calibration

Reduce bootstrap iterations for speed:

```python
framework_config = FrameworkConfig(
    bootstrap_iterations=500,  # Instead of 10_000
)
```

## Running Tests with Ollama

The test suite includes end-to-end tests that use Ollama:

```bash
# Tests automatically skip if Ollama not available
pytest tests/test_end_to_end_ollama.py -v

# Run all tests (Ollama tests skip gracefully)
pytest tests/ -v
```

## Troubleshooting

### "Ollama not available"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (if installed)
# macOS: Open the Ollama app
# Linux: ollama serve
```

### "Model not available"

```bash
# List installed models
ollama list

# Pull the model
ollama pull qwen2.5:3b
```

### Slow performance

```bash
# Use a smaller model
ollama pull qwen2.5:1.5b

# Or reduce context windows in FrameworkConfig
T1=4_000, T2=16_000
```

### Out of memory

```bash
# Use a smaller model
ollama pull qwen2.5:1.5b

# Or reduce batch size
K=3  # Sample fewer documents per instance
```

## Model Recommendations

### For Testing (Fast)
- `qwen2.5:1.5b` - Fastest, basic quality
- `qwen2.5:3b` - **Recommended** - Good balance

### For Development (Quality)
- `qwen2.5:7b` - Better summaries, slower
- `llama3.2:3b` - Alternative, similar to qwen2.5:3b

### For Production
- Use Anthropic Claude Sonnet via API
- Ollama is for development/testing only

## Next Steps

1. **Test locally** with Ollama on small corpus
2. **Refine prompts** based on Ollama results
3. **Switch to Claude** for production runs
4. **Use Batch API** for 50% cost savings

The prompt templates work the same for both Ollama and Claude, so you can develop locally and deploy to production seamlessly.
