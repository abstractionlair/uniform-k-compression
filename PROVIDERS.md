# Multi-Provider Support

The Uniform-K Compression framework now supports multiple LLM providers with automatic caching and batch API support where available.

## Supported Providers

### ✅ Anthropic Claude (Default)
- **Models:** Opus 4.1, Sonnet 4.5, Haiku 4.5
- **Batch API:** ✅ Yes (50% discount)
- **Caching:** ✅ Yes (up to 90% reduction, 5-min TTL)
- **Special Features:** Batch + caching discounts stack!
- **API Key:** `ANTHROPIC_API_KEY`

### ✅ OpenAI GPT
- **Models:** GPT-5.1, GPT-5.1 Instant, GPT-4o, o1 family
- **Batch API:** ✅ Yes (50% discount on input AND output)
- **Caching:** ✅ Yes (automatic on prompts >1,024 tokens, 50% discount)
- **API Key:** `OPENAI_API_KEY`
- **Install:** `pip install openai`

### ✅ Google Gemini
- **Models:** Gemini 3 Pro, 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite
- **Batch API:** ✅ Yes (50% discount, up to 200K requests/batch)
- **Caching:** ✅ Yes (implicit: automatic 90% discount on 2.5 models)
- **Note:** Batch doesn't support explicit caching (only implicit)
- **API Key:** `GOOGLE_API_KEY`
- **Install:** `pip install google-generativeai`

### ✅ xAI Grok
- **Models:** Grok 4, Grok 4.1 Fast (reasoning/non-reasoning), Grok 3/Mini
- **Batch API:** ⚠️ Not documented
- **Caching:** ✅ Yes (automatic, >90% hit rates, very cheap at $0.02/MTok)
- **API Key:** `XAI_API_KEY`
- **Install:** `pip install openai` (uses OpenAI SDK)

## Installation

### Core (Anthropic only)
```bash
pip install -r requirements.txt
```

### With All Providers
```bash
pip install -r requirements.txt
pip install openai google-generativeai
```

### Individual Providers
```bash
# For OpenAI
pip install openai

# For Google Gemini
pip install google-generativeai

# For xAI Grok (uses OpenAI SDK)
pip install openai
```

## Quick Start

### Using FrameworkConfig

```python
from core import FractalSummarizer, FrameworkConfig

# Anthropic (default)
config = FrameworkConfig(
    provider='anthropic',
    model='sonnet',
    large_context_model='sonnet[1m]',
    use_batch_api=False  # Set to True for 50% savings
)

# OpenAI
config = FrameworkConfig(
    provider='openai',
    model='gpt-5.1',
    use_batch_api=False
)

# Google Gemini
config = FrameworkConfig(
    provider='google',  # or 'gemini'
    model='gemini-2.5-flash',
    large_context_model='gemini-2.5-pro'
)

# xAI Grok
config = FrameworkConfig(
    provider='xai',  # or 'grok'
    model='grok-4.1-fast',  # or 'grok-4' for deeper reasoning
)

summarizer = FractalSummarizer(config)
```

### Using Provider Factory Directly

```python
from core import create_provider

# Create a provider instance
provider = create_provider(
    provider_name='anthropic',
    model='sonnet',
    large_context_model='sonnet[1m]',
    temperature=1.0
)

# Use the provider
output, tokens, usage = provider.call(
    prompt="Summarize this text...",
    context_size="small",
    max_tokens=50_000
)

# Calculate cost
cost = provider.calculate_cost(usage, 'sonnet')
print(f"Cost: ${cost:.4f}")
```

## Model Selection Guide

### Anthropic
- **Haiku 4.5:** Fast, high-volume processing
- **Sonnet 4.5:** Balanced performance (recommended default)
- **Opus 4.1:** Hardest tasks, best quality

### OpenAI
- **GPT-5.1:** General purpose flagship
- **GPT-5.1 Instant:** More conversational
- **GPT-5.1 Thinking:** Best reasoning
- **GPT-4o:** Cost-effective alternative

### Google
- **Gemini 3 Pro:** Best quality, multimodal
- **Gemini 2.5 Flash:** Best price-performance (recommended)
- **Gemini 2.5 Flash-Lite:** Ultra-fast, lightweight

### xAI
- **Grok 4.1 Fast:** Production speed (recommended)
- **Grok 4:** Deeper reasoning (slower, requires high max_tokens)

## Batch API Usage

Batch API provides 50% cost savings but takes hours instead of minutes.

```python
config = FrameworkConfig(
    provider='anthropic',  # or 'openai', 'google'
    model='sonnet',
    use_batch_api=True,
    batch_poll_interval=300  # Check every 5 minutes
)
```

**Provider Support:**
- ✅ Anthropic: Full support, can stack with caching
- ✅ OpenAI: Supported (file-based, not fully implemented)
- ✅ Google: Supported (Vertex AI required, not fully implemented)
- ❌ xAI: Not documented

## Caching

All providers support automatic caching with different approaches:

### Anthropic
- **Type:** Explicit prompt caching
- **Discount:** Up to 90%
- **TTL:** 5 minutes
- **Requirements:** Model-specific token minimums (4,096 for Haiku 4.5)

### OpenAI
- **Type:** Automatic
- **Discount:** 50%
- **TTL:** 5-10 minutes, forced eviction at 1 hour
- **Requirements:** Prompts >1,024 tokens

### Google
- **Type:** Implicit (automatic) + Explicit
- **Discount:** 90% (Gemini 2.5), 75% (Gemini 2.0)
- **Requirements:** Min 2,048 tokens for explicit caching
- **Note:** Implicit caching is automatic, no setup needed

### xAI
- **Type:** Automatic
- **Discount:** Very cheap ($0.02/MTok for cached tokens)
- **Hit Rate:** >90% on code model
- **Note:** Always enabled

## Architecture

The provider system is built on a clean abstraction:

```
BaseProvider (abstract)
├── AnthropicProvider
├── OpenAIProvider
├── GoogleProvider
└── XAIProvider
```

All providers implement:
- `call(prompt, context_size, ...)` - Make an API call
- `supports_batch()` - Check batch API support
- `supports_caching()` - Check caching support
- `calculate_cost(usage, model)` - Calculate costs
- `get_total_usage()` - Get cumulative usage stats

Batch API methods (if supported):
- `submit_batch(requests)` - Submit batch
- `check_batch_status(batch_id)` - Check status
- `wait_for_batch(batch_id)` - Wait for completion
- `get_batch_results(batch_id)` - Get results

## Environment Variables

Set the appropriate API key for your provider:

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
export GOOGLE_API_KEY='AI...'
export XAI_API_KEY='xai-...'
```

## Examples

See `examples/provider_comparison.py` for a complete demonstration of configuring and using each provider.

## Cost Optimization

**Best Practices:**
1. **Use batch API** when time-to-result isn't critical (50% savings)
2. **Enable caching** for repeated content (automatic on most providers)
3. **Choose the right model:**
   - Development/testing: Haiku (Anthropic), GPT-4o mini (OpenAI), Gemini 2.5 Flash-Lite (Google)
   - Production: Sonnet (Anthropic), GPT-5.1 (OpenAI), Gemini 2.5 Flash (Google)
   - Complex tasks: Opus (Anthropic), GPT-5.1 Thinking (OpenAI), Gemini 3 Pro (Google)

**Discount Stacking:**
- Anthropic: Batch (50%) + Caching (90%) = up to 95% total savings
- OpenAI: Batch (50%) + Caching (50%) = up to 75% total savings
- Google: Batch (50%) + Implicit Caching (90%) = up to 95% total savings

## Backward Compatibility

The old `LLMInterface` and `BatchInterface` classes are still available for backward compatibility but are deprecated. New code should use the provider system via `FrameworkConfig(provider='...')` or `create_provider()`.

## Testing

Run the test suite to verify provider functionality:

```bash
python test_providers.py
```

This will check:
- Provider factory instantiation
- Capability detection
- Configuration validation
- Provider availability

## Troubleshooting

**Import Error: "OpenAI SDK not installed"**
```bash
pip install openai
```

**Import Error: "Google Generative AI SDK not installed"**
```bash
pip install google-generativeai
```

**ValueError: "Unknown provider"**
- Check that the provider is installed
- Run `python test_providers.py` to see available providers

**ValueError: "Provider does not support batch API"**
- xAI doesn't support batch API
- For OpenAI/Google, batch API implementation may need completion

## Future Enhancements

- Complete OpenAI batch API file-based implementation
- Complete Google Gemini batch API (Vertex AI)
- Add support for Azure OpenAI
- Add support for AWS Bedrock
- Per-provider cost tracking and optimization
