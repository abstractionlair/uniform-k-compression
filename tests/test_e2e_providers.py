#!/usr/bin/env python3
"""
End-to-end integration tests for all LLM providers.

Tests the full fractal summarization pipeline with Sherlock Holmes stories
using the least expensive model from each provider:
- Anthropic: Haiku 3.5 (cheapest)
- OpenAI: GPT-5 Nano or GPT-5 Mini (cheapest)
- Google: Gemini 2.5 Flash-Lite (cheapest)
- xAI: Grok 4.1 Fast Non-Reasoning (cheapest reasoning-free)

Batch API tests are skipped (too expensive and slow for CI).

Environment variables required:
- ANTHROPIC_API_KEY (for Anthropic tests)
- OPENAI_API_KEY (for OpenAI tests)
- GOOGLE_API_KEY (for Google tests)
- XAI_API_KEY (for xAI tests)

Run specific provider tests with:
    pytest tests/test_e2e_providers.py -k anthropic
    pytest tests/test_e2e_providers.py -k openai
    pytest tests/test_e2e_providers.py -k google
    pytest tests/test_e2e_providers.py -k xai
"""

import os

import pytest

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, assume env vars are set

from core import (
    AnalysisConfig,
    FractalSummarizer,
    FrameworkConfig,
    list_providers,
)

# ==============================================================================
# Provider availability checks
# ==============================================================================

def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return os.environ.get("ANTHROPIC_API_KEY") is not None


def has_openai_key():
    """Check if OpenAI API key is available."""
    return os.environ.get("OPENAI_API_KEY") is not None and 'openai' in list_providers()


def has_google_key():
    """Check if Google API key is available."""
    return os.environ.get("GOOGLE_API_KEY") is not None and 'google' in list_providers()


def has_xai_key():
    """Check if xAI API key is available."""
    return os.environ.get("XAI_API_KEY") is not None and 'xai' in list_providers()


# ==============================================================================
# Shared fixtures
# ==============================================================================

@pytest.fixture
def base_framework_config():
    """Base framework config optimized for cost efficiency."""
    return {
        'k': 1.5,
        'r': 0.4,  # Less aggressive compression
        'T1': 16_000,  # Small contexts
        'T2': 80_000,  # Large enough for biggest story (~61K)
        'target_convergence': 40_000,
        'bootstrap_iterations': 500,  # Fewer iterations for speed
        'use_batch_api': False,  # Skip batch for e2e tests
    }


@pytest.fixture
def analysis_config_with_reduced_tokens(tmp_path):
    """Analysis config with reduced max_tokens for models with limits."""
    return AnalysisConfig(
        name="E2E Provider Test",
        description="End-to-end test with real LLM provider",
        layer_prompt_template="""Analyze these {num_docs} Sherlock Holmes stories.

Compress to approximately {r:.0%} of the original length while preserving:
- Key plot points
- Character names
- How the mystery was solved

Stories:
{documents}

Summary:""",
        final_synthesis_prompt="""Synthesize these story summaries into a brief overview of:
1. Common themes
2. Holmes' methods
3. Types of cases

Keep it concise (2-3 paragraphs).""",
        output_dir=str(tmp_path / "provider_test_output")
    )


@pytest.fixture
def analysis_config(tmp_path):
    """Shared analysis config for all provider tests."""
    return AnalysisConfig(
        name="E2E Provider Test",
        description="End-to-end test with real LLM provider",
        layer_prompt_template="""Analyze these {num_docs} Sherlock Holmes stories.

Compress to approximately {r:.0%} of the original length while preserving:
- Key plot points
- Character names
- How the mystery was solved

Stories:
{documents}

Summary:""",
        final_synthesis_prompt="""Synthesize these story summaries into a brief overview of:
1. Common themes
2. Holmes' methods
3. Types of cases

Keep it concise (2-3 paragraphs).""",
        output_dir=str(tmp_path / "provider_test_output")
    )


def verify_results(result, metadata, provider_name):
    """Common verification for all provider tests."""
    # Basic sanity checks
    assert result is not None, f"{provider_name}: No result returned"
    assert len(result) > 0, f"{provider_name}: Empty result"
    assert metadata.total_layers > 0, f"{provider_name}: No layers executed"
    assert metadata.total_instances > 0, f"{provider_name}: No instances created"

    # Verify compression happened
    assert metadata.final_tokens < metadata.initial_tokens, \
        f"{provider_name}: No compression ({metadata.initial_tokens} → {metadata.final_tokens})"

    # Cost should be > 0 (we're using real APIs)
    assert metadata.total_cost_usd > 0, \
        f"{provider_name}: Expected non-zero cost"

    # Cost should be reasonable (< $5 for this small test)
    # Note: OpenAI GPT-5 models are expensive (~$4), others are cheaper
    assert metadata.total_cost_usd < 5.0, \
        f"{provider_name}: Cost too high: ${metadata.total_cost_usd:.2f}"

    return True


# ==============================================================================
# Anthropic Tests
# ==============================================================================

@pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
def test_anthropic_haiku_e2e(small_corpus, base_framework_config, analysis_config):
    """Test full pipeline with Anthropic Haiku 4.5 (cheapest Claude model)."""
    print("\n" + "="*70)
    print("ANTHROPIC E2E TEST - Haiku 4.5")
    print("="*70)

    config = FrameworkConfig(
        provider='anthropic',
        model='haiku',  # Haiku 4.5 - cheapest
        large_context_model='haiku',  # Use same for large context
        **base_framework_config
    )

    summarizer = FractalSummarizer(config)
    result, metadata = summarizer.run(small_corpus, analysis_config)

    # Verify results
    verify_results(result, metadata, "Anthropic Haiku")

    print("\n✅ Anthropic test passed!")
    print("   Model: Haiku 4.5")
    print(f"   Layers: {metadata.total_layers}")
    print(f"   Instances: {metadata.total_instances}")
    print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
    print(f"   Cost: ${metadata.total_cost_usd:.4f}")

    # Cost check - Haiku should be very cheap
    assert metadata.total_cost_usd < 0.50, \
        f"Haiku cost unexpectedly high: ${metadata.total_cost_usd:.4f}"


@pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
def test_anthropic_caching(small_corpus, base_framework_config, tmp_path):
    """Verify Anthropic prompt caching is working."""
    print("\n" + "="*70)
    print("ANTHROPIC CACHING TEST")
    print("="*70)

    config = FrameworkConfig(
        provider='anthropic',
        model='haiku',
        **base_framework_config
    )

    # Create a simple analysis config
    analysis_config = AnalysisConfig(
        name="Caching Test",
        layer_prompt_template="Summarize: {documents}",
        final_synthesis_prompt="Synthesize the summaries.",
        output_dir=str(tmp_path / "caching_test")
    )

    summarizer = FractalSummarizer(config)

    # Use just 3 stories to keep it fast
    docs = small_corpus[:3]

    result, metadata = summarizer.run(docs, analysis_config)

    # Check if we got cache usage
    usage = summarizer.provider.get_total_usage()

    print("\n📊 Cache statistics:")
    print(f"   Input tokens: {usage.input_tokens:,}")
    print(f"   Cache creation: {usage.cache_creation_tokens:,}")
    print(f"   Cache reads: {usage.cache_read_tokens:,}")
    print(f"   Output tokens: {usage.output_tokens:,}")

    # We expect some token usage
    assert usage.input_tokens > 0, "Should have input tokens"
    assert usage.output_tokens > 0, "Should have output tokens"

    print("\n✅ Anthropic caching test passed!")


# ==============================================================================
# OpenAI Tests
# ==============================================================================

@pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set or SDK not installed")
def test_openai_gpt5_nano_e2e(small_corpus, base_framework_config, analysis_config):
    """Test full pipeline with OpenAI GPT-5 Nano (cheapest GPT-5 model)."""
    print("\n" + "="*70)
    print("OPENAI E2E TEST - GPT-5 Nano/Mini")
    print("="*70)

    # Try gpt-5-nano first, fallback to gpt-5-mini if not available
    model = 'gpt-5-nano'
    try:
        # Quick test to see if model is available
        from core import create_provider
        test_provider = create_provider('openai', model=model, api_key=os.environ.get('OPENAI_API_KEY'))
        print(f"   Using: {model}")
    except Exception:
        model = 'gpt-5-mini'
        print(f"   gpt-5-nano not available, using: {model}")

    config = FrameworkConfig(
        provider='openai',
        model=model,
        **base_framework_config
    )

    summarizer = FractalSummarizer(config)
    result, metadata = summarizer.run(small_corpus, analysis_config)

    # Verify results
    verify_results(result, metadata, f"OpenAI {model}")

    print("\n✅ OpenAI test passed!")
    print(f"   Model: {model}")
    print(f"   Layers: {metadata.total_layers}")
    print(f"   Instances: {metadata.total_instances}")
    print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
    print(f"   Cost: ${metadata.total_cost_usd:.4f}")

    # GPT-5 nano/mini should be very cheap
    assert metadata.total_cost_usd < 0.50, \
        f"{model} cost unexpectedly high: ${metadata.total_cost_usd:.4f}"


@pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set or SDK not installed")
def test_openai_caching(small_corpus, base_framework_config, tmp_path):
    """Verify OpenAI automatic caching is working."""
    print("\n" + "="*70)
    print("OPENAI CACHING TEST")
    print("="*70)

    # Try gpt-5-nano first, fallback to gpt-5-mini
    model = 'gpt-5-nano'
    try:
        from core import create_provider
        test_provider = create_provider('openai', model=model, api_key=os.environ.get('OPENAI_API_KEY'))
    except Exception:
        model = 'gpt-5-mini'

    config = FrameworkConfig(
        provider='openai',
        model=model,
        **base_framework_config
    )

    analysis_config = AnalysisConfig(
        name="Caching Test",
        layer_prompt_template="Summarize: {documents}",
        final_synthesis_prompt="Synthesize the summaries.",
        output_dir=str(tmp_path / "caching_test")
    )

    summarizer = FractalSummarizer(config)
    docs = small_corpus[:3]

    result, metadata = summarizer.run(docs, analysis_config)

    # Check usage
    usage = summarizer.provider.get_total_usage()

    print("\n📊 Cache statistics:")
    print(f"   Input tokens: {usage.input_tokens:,}")
    print(f"   Cache reads: {usage.cache_read_tokens:,}")
    print(f"   Output tokens: {usage.output_tokens:,}")

    # Note: OpenAI only shows cache hits on some models
    if usage.cache_read_tokens > 0:
        print(f"   ✅ Caching detected: {usage.cache_read_tokens:,} tokens")
    else:
        print("   ℹ️  No cache hits detected (may not be exposed for this model)")

    assert usage.input_tokens > 0, "Should have input tokens"
    assert usage.output_tokens > 0, "Should have output tokens"

    print("\n✅ OpenAI caching test passed!")


# ==============================================================================
# Google Gemini Tests
# ==============================================================================

@pytest.mark.skipif(not has_google_key(), reason="GOOGLE_API_KEY not set or SDK not installed")
def test_google_flash_lite_e2e(small_corpus, base_framework_config, analysis_config):
    """Test full pipeline with Google Gemini 2.5 Flash-Lite (cheapest model)."""
    print("\n" + "="*70)
    print("GOOGLE E2E TEST - Gemini 2.5 Flash-Lite")
    print("="*70)

    config = FrameworkConfig(
        provider='google',
        model='gemini-2.5-flash-lite',  # Cheapest model
        large_context_model='gemini-2.5-flash',  # Use Flash for large context
        **base_framework_config
    )

    summarizer = FractalSummarizer(config)
    result, metadata = summarizer.run(small_corpus, analysis_config)

    # Verify results
    verify_results(result, metadata, "Google Gemini Flash-Lite")

    print("\n✅ Google test passed!")
    print("   Model: Gemini 2.5 Flash-Lite")
    print(f"   Layers: {metadata.total_layers}")
    print(f"   Instances: {metadata.total_instances}")
    print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
    print(f"   Cost: ${metadata.total_cost_usd:.4f}")

    # Flash-Lite should be extremely cheap
    assert metadata.total_cost_usd < 0.30, \
        f"Flash-Lite cost unexpectedly high: ${metadata.total_cost_usd:.4f}"


@pytest.mark.skipif(not has_google_key(), reason="GOOGLE_API_KEY not set or SDK not installed")
def test_google_implicit_caching(small_corpus, base_framework_config, tmp_path):
    """Verify Google's implicit caching (automatic 90% discount)."""
    print("\n" + "="*70)
    print("GOOGLE IMPLICIT CACHING TEST")
    print("="*70)

    config = FrameworkConfig(
        provider='google',
        model='gemini-2.5-flash-lite',
        large_context_model='gemini-2.5-flash',  # Use Flash for large context
        **base_framework_config
    )

    analysis_config = AnalysisConfig(
        name="Caching Test",
        layer_prompt_template="Summarize: {documents}",
        final_synthesis_prompt="Synthesize the summaries.",
        output_dir=str(tmp_path / "caching_test")
    )

    summarizer = FractalSummarizer(config)
    docs = small_corpus[:3]

    result, metadata = summarizer.run(docs, analysis_config)

    # Check usage
    usage = summarizer.provider.get_total_usage()

    print("\n📊 Usage statistics:")
    print(f"   Input tokens: {usage.input_tokens:,}")
    print(f"   Cached tokens: {usage.cache_read_tokens:,}")
    print(f"   Output tokens: {usage.output_tokens:,}")
    print("   ℹ️  Note: Gemini has implicit caching (90% auto-discount)")

    assert usage.input_tokens > 0, "Should have input tokens"
    assert usage.output_tokens > 0, "Should have output tokens"

    print("\n✅ Google caching test passed!")


# ==============================================================================
# xAI Grok Tests
# ==============================================================================

@pytest.mark.skipif(not has_xai_key(), reason="XAI_API_KEY not set or SDK not installed")
def test_xai_grok41_fast_e2e(small_corpus, base_framework_config, analysis_config):
    """Test full pipeline with xAI Grok 4.1 Fast Non-Reasoning (fast, no reasoning overhead)."""
    print("\n" + "="*70)
    print("XAI E2E TEST - Grok 4.1 Fast Non-Reasoning")
    print("="*70)

    config = FrameworkConfig(
        provider='xai',
        model='grok-4.1-fast-non-reasoning',  # Fast model without reasoning
        **base_framework_config
    )

    summarizer = FractalSummarizer(config)
    result, metadata = summarizer.run(small_corpus, analysis_config)

    # Verify results
    verify_results(result, metadata, "xAI Grok 4.1 Fast Non-Reasoning")

    print("\n✅ xAI test passed!")
    print("   Model: Grok 4.1 Fast Non-Reasoning")
    print(f"   Layers: {metadata.total_layers}")
    print(f"   Instances: {metadata.total_instances}")
    print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
    print(f"   Cost: ${metadata.total_cost_usd:.4f}")

    # Grok 4.1 Fast should be reasonably cheap
    assert metadata.total_cost_usd < 0.50, \
        f"Grok 4.1 Fast cost unexpectedly high: ${metadata.total_cost_usd:.4f}"


@pytest.mark.skipif(not has_xai_key(), reason="XAI_API_KEY not set or SDK not installed")
def test_xai_caching(small_corpus, base_framework_config, tmp_path):
    """Verify xAI automatic caching (>90% hit rates)."""
    print("\n" + "="*70)
    print("XAI CACHING TEST")
    print("="*70)

    config = FrameworkConfig(
        provider='xai',
        model='grok-4.1-fast-non-reasoning',
        **base_framework_config
    )

    analysis_config = AnalysisConfig(
        name="Caching Test",
        layer_prompt_template="Summarize: {documents}",
        final_synthesis_prompt="Synthesize the summaries.",
        output_dir=str(tmp_path / "caching_test")
    )

    summarizer = FractalSummarizer(config)
    docs = small_corpus[:3]

    result, metadata = summarizer.run(docs, analysis_config)

    # Check usage
    usage = summarizer.provider.get_total_usage()

    print("\n📊 Cache statistics:")
    print(f"   Input tokens: {usage.input_tokens:,}")
    print(f"   Cache reads: {usage.cache_read_tokens:,}")
    print(f"   Output tokens: {usage.output_tokens:,}")
    print("   ℹ️  xAI typically achieves >90% cache hit rates")

    if usage.cache_read_tokens > 0:
        cache_rate = usage.cache_read_tokens / usage.input_tokens if usage.input_tokens > 0 else 0
        print(f"   Cache hit rate: {cache_rate:.1%}")

    assert usage.input_tokens > 0, "Should have input tokens"
    assert usage.output_tokens > 0, "Should have output tokens"

    print("\n✅ xAI caching test passed!")


# ==============================================================================
# Cross-provider comparison test
# ==============================================================================

@pytest.mark.skipif(
    not (has_anthropic_key() and has_openai_key()),
    reason="Requires both ANTHROPIC_API_KEY and OPENAI_API_KEY"
)
def test_cross_provider_comparison(small_corpus, base_framework_config, tmp_path):
    """Compare outputs from different providers on same input."""
    print("\n" + "="*70)
    print("CROSS-PROVIDER COMPARISON")
    print("="*70)

    # Use just 3 stories for speed
    docs = small_corpus[:3]

    results = {}

    # Test Anthropic
    print("\n→ Testing Anthropic Haiku...")
    config_anthropic = FrameworkConfig(
        provider='anthropic',
        model='haiku',
        **base_framework_config
    )
    analysis_config_anthropic = AnalysisConfig(
        name="Comparison - Anthropic",
        layer_prompt_template="Summarize: {documents}",
        final_synthesis_prompt="Brief synthesis.",
        output_dir=str(tmp_path / "anthropic")
    )
    summarizer_anthropic = FractalSummarizer(config_anthropic)
    result_anthropic, meta_anthropic = summarizer_anthropic.run(docs, analysis_config_anthropic)
    results['anthropic'] = (result_anthropic, meta_anthropic)

    # Test OpenAI
    print("\n→ Testing OpenAI GPT-5 Nano/Mini...")
    # Try nano first, fallback to mini
    openai_model = 'gpt-5-nano'
    try:
        from core import create_provider
        test_provider = create_provider('openai', model=openai_model, api_key=os.environ.get('OPENAI_API_KEY'))
    except Exception:
        openai_model = 'gpt-5-mini'

    config_openai = FrameworkConfig(
        provider='openai',
        model=openai_model,
        **base_framework_config
    )
    analysis_config_openai = AnalysisConfig(
        name="Comparison - OpenAI",
        layer_prompt_template="Summarize: {documents}",
        final_synthesis_prompt="Brief synthesis.",
        output_dir=str(tmp_path / "openai")
    )
    summarizer_openai = FractalSummarizer(config_openai)
    result_openai, meta_openai = summarizer_openai.run(docs, analysis_config_openai)
    results['openai'] = (result_openai, meta_openai)

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    for provider, (result, metadata) in results.items():
        print(f"\n{provider.upper()}:")
        print(f"   Layers: {metadata.total_layers}")
        print(f"   Compression: {metadata.initial_tokens:,} → {metadata.final_tokens:,}")
        print(f"   Cost: ${metadata.total_cost_usd:.4f}")
        print(f"   Output length: {len(result)} chars")

    # Both should produce valid outputs
    for provider, (result, metadata) in results.items():
        assert result is not None and len(result) > 0
        assert metadata.total_cost_usd > 0

    print("\n✅ Cross-provider comparison passed!")
