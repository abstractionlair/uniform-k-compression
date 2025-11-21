# End-to-End Provider Tests

Comprehensive e2e tests for all LLM providers using Sherlock Holmes stories.

## Test Coverage

### Providers Tested
- ✅ **Anthropic** (Haiku 4.5 - cheapest)
- ✅ **OpenAI** (GPT-4o mini - cheapest)
- ✅ **Google** (Gemini 2.5 Flash-Lite - cheapest)
- ✅ **xAI** (Grok 3 Mini - cheapest)

### Features Tested
- Full fractal summarization pipeline
- Multiple layers of compression
- Prompt caching (where available)
- Cost tracking and optimization
- Cross-provider comparison

### Not Tested (By Design)
- ❌ Batch API (too expensive, takes hours)
- ❌ Most expensive models (keeping costs down)

## Prerequisites

### 1. Install Dependencies

```bash
# Core (Anthropic)
pip install -r requirements.txt

# All providers
pip install openai google-generativeai
```

### 2. Set API Keys

```bash
# Set the API keys for providers you want to test
export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
export GOOGLE_API_KEY='AI...'
export XAI_API_KEY='xai-...'
```

### 3. Verify Test Data

Tests use Sherlock Holmes stories from `tests/test_data/stories/`.
Make sure these files exist (they should be in the repository).

## Running Tests

### Run All Available Provider Tests

```bash
pytest tests/test_e2e_providers.py -v
```

This will run tests for all providers that have API keys set.
Providers without keys will be skipped.

### Run Specific Provider Tests

```bash
# Anthropic only
pytest tests/test_e2e_providers.py -k anthropic -v

# OpenAI only
pytest tests/test_e2e_providers.py -k openai -v

# Google only
pytest tests/test_e2e_providers.py -k google -v

# xAI only
pytest tests/test_e2e_providers.py -k xai -v
```

### Run Specific Test Types

```bash
# E2E tests only (skip caching tests)
pytest tests/test_e2e_providers.py -k "e2e" -v

# Caching tests only
pytest tests/test_e2e_providers.py -k "caching" -v

# Cross-provider comparison
pytest tests/test_e2e_providers.py -k "comparison" -v
```

### Run with Output

```bash
# Show detailed output
pytest tests/test_e2e_providers.py -v -s

# Show only test names
pytest tests/test_e2e_providers.py -v --tb=no
```

## Expected Costs

All tests use the cheapest models and small corpus (5 stories).

**Estimated cost per provider:**
- Anthropic Haiku: ~$0.10-0.30
- OpenAI GPT-4o mini: ~$0.05-0.20
- Google Gemini Flash-Lite: ~$0.02-0.10
- xAI Grok 3 Mini: ~$0.05-0.20

**Total for all providers:** ~$0.25-0.80

Tests include cost assertions to prevent unexpectedly high charges:
- Individual tests capped at $0.50
- Haiku test capped at $0.50
- GPT-4o mini test capped at $0.50
- Flash-Lite test capped at $0.30
- Grok 3 Mini test capped at $0.50

## Test Structure

Each provider has two main tests:

1. **E2E Pipeline Test** (`test_<provider>_<model>_e2e`)
   - Runs full fractal summarization
   - Verifies compression
   - Checks output quality
   - Validates cost

2. **Caching Test** (`test_<provider>_caching`)
   - Verifies caching is working
   - Checks usage statistics
   - Confirms cost optimization

## Common Issues

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY='your-key-here'
```

### "OpenAI SDK not installed"
```bash
pip install openai
```

### "Google Generative AI SDK not installed"
```bash
pip install google-generativeai
```

### Test Data Missing
Make sure Sherlock Holmes stories exist in `tests/test_data/stories/`.
These should be included in the repository.

### High Costs
If a test fails with "cost too high", check:
1. Are you using the correct (cheap) models?
2. Is the test corpus limited to 5 stories?
3. Are you accidentally using batch API?

## Test Output Example

```
======================================================================
ANTHROPIC E2E TEST - Haiku 4.5
======================================================================

Layer 1: 5 docs → 3 instances (K=3, 0 large context, 0.42x compression)
Layer 2: 3 docs → 2 instances (K=2, 0 large context, 0.38x compression)
Final synthesis: 1,234 tokens

✅ Anthropic test passed!
   Model: Haiku 4.5
   Layers: 2
   Instances: 5
   Compression: 45,678 → 3,456
   Cost: $0.1234
```

## Continuous Integration

These tests are designed to run in CI with minimal cost:

```yaml
# .github/workflows/test.yml
- name: Run E2E Tests
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    pytest tests/test_e2e_providers.py -v
```

Tests automatically skip providers without API keys.

## Debugging

### Run with Python Debugger
```bash
pytest tests/test_e2e_providers.py::test_anthropic_haiku_e2e -v -s --pdb
```

### Check Provider Availability
```python
from core import list_providers
print(list_providers())  # ['anthropic', 'openai', 'google', 'xai']
```

### Verify API Keys
```bash
python test_providers.py
```

## Adding New Provider Tests

To add tests for a new provider:

1. Add provider check function:
```python
def has_newprovider_key():
    return os.environ.get("NEWPROVIDER_API_KEY") is not None
```

2. Add e2e test:
```python
@pytest.mark.skipif(not has_newprovider_key(), reason="...")
def test_newprovider_e2e(small_corpus, base_framework_config, analysis_config):
    config = FrameworkConfig(provider='newprovider', model='cheap-model', ...)
    # ... rest of test
```

3. Add caching test (if provider supports caching):
```python
@pytest.mark.skipif(not has_newprovider_key(), reason="...")
def test_newprovider_caching(...):
    # ... caching verification
```

## Related Files

- `test_e2e_providers.py` - Main e2e tests
- `test_end_to_end_ollama.py` - Free local testing with Ollama
- `conftest.py` - Shared fixtures
- `test_data/stories/` - Sherlock Holmes test corpus

## See Also

- `../PROVIDERS.md` - Provider documentation
- `../examples/provider_comparison.py` - Provider usage examples
- `../test_providers.py` - Provider availability check
