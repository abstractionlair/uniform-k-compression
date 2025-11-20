# Examples

This directory contains working examples demonstrating how to use fractal summarization.

## Quick Start Example

The simplest way to get started:

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Run the example
python examples/run_simple_example.py
```

This will:
1. Load 5 Sherlock Holmes stories from `sample_data/`
2. Run fractal summarization with conservative settings
3. Generate a comprehensive analysis
4. Save results to `examples/output/`

**Cost**: Approximately $0.50-1.00 depending on compression achieved.

## What's Included

- `sample_data/` - 5 public domain Sherlock Holmes stories (~5,000 lines total)
- `example_config.json` - Example analysis configuration
- `run_simple_example.py` - Complete runnable example with API

## Using Your Own Data

To analyze your own documents:

1. Create a directory with your `.txt` files
2. Copy and modify `example_config.json`
3. Use `run_analysis.py` from project root:

```bash
python run_analysis.py \
  --input /path/to/your/documents \
  --analysis-config examples/example_config.json \
  --output output/my_analysis
```

Or write your own Python script following the pattern in `run_simple_example.py`.

## Next Steps

See the main [README.md](../README.md) for:
- Detailed explanation of algorithm parameters (k, r, T1, T2)
- Cost optimization strategies
- Batch API usage for 50% cost savings
- Preprocessing utilities for large documents
