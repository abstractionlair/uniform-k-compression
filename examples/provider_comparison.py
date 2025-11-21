#!/usr/bin/env python3
"""
Example: Using different LLM providers with the fractal summarization framework.

This example shows how to configure and use different providers:
- Anthropic Claude
- OpenAI GPT
- Google Gemini
- xAI Grok

Each provider can be selected via the FrameworkConfig.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import FrameworkConfig, create_provider, list_providers


def show_available_providers():
    """Display available providers and their capabilities."""
    print("=" * 70)
    print("AVAILABLE PROVIDERS")
    print("=" * 70)

    providers = list_providers()
    print(f"\nInstalled providers: {', '.join(providers)}\n")

    from core import get_provider_info
    info = get_provider_info()

    for name, caps in info.items():
        print(f"{name.upper()}:")
        print(f"  Batch API: {'✓' if caps['supports_batch'] else '✗'}")
        print(f"  Caching:   {'✓' if caps['supports_caching'] else '✗'}")
        print()


def example_anthropic():
    """Example using Anthropic Claude."""
    print("=" * 70)
    print("ANTHROPIC CLAUDE EXAMPLE")
    print("=" * 70)

    config = FrameworkConfig(
        provider='anthropic',
        model='sonnet',
        large_context_model='sonnet[1m]',
        k=1.5,
        r=0.3,
        temperature=1.0,
        use_batch_api=False  # Set to True for 50% cost savings
    )

    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Large context model: {config.large_context_model}")
    print(f"Batch API: {config.use_batch_api}")
    print()

    # Create provider instance (using dummy API key for demo)
    provider = create_provider(
        provider_name=config.provider,
        model=config.model,
        large_context_model=config.large_context_model,
        temperature=config.temperature,
        api_key='dummy-key-for-demo'  # Replace with real API key
    )

    print(f"✓ Created {provider.get_provider_name()} provider")
    print(f"  Supports batch: {provider.supports_batch()}")
    print(f"  Supports caching: {provider.supports_caching()}")
    print()


def example_openai():
    """Example using OpenAI GPT."""
    print("=" * 70)
    print("OPENAI GPT EXAMPLE")
    print("=" * 70)

    config = FrameworkConfig(
        provider='openai',
        model='gpt-5.1',  # Or gpt-5.1-instant, gpt-4o, etc.
        k=1.5,
        r=0.3,
        temperature=1.0,
        use_batch_api=False
    )

    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print("Note: OpenAI models have large context by default")
    print()

    try:
        provider = create_provider(
            provider_name=config.provider,
            model=config.model,
            temperature=config.temperature,
            api_key='dummy-key-for-demo'
        )
        print(f"✓ Created {provider.get_provider_name()} provider")
        print(f"  Supports batch: {provider.supports_batch()}")
        print(f"  Supports caching: {provider.supports_caching()}")
    except Exception as e:
        print(f"⚠️  OpenAI provider not available: {e}")
        print("   Install with: pip install openai")
    print()


def example_google():
    """Example using Google Gemini."""
    print("=" * 70)
    print("GOOGLE GEMINI EXAMPLE")
    print("=" * 70)

    config = FrameworkConfig(
        provider='google',  # or 'gemini'
        model='gemini-2.5-flash',
        large_context_model='gemini-2.5-pro',
        k=1.5,
        r=0.3,
        temperature=1.0,
        use_batch_api=False
    )

    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Large context model: {config.large_context_model}")
    print("Note: Gemini has automatic implicit caching (90% discount)")
    print()

    try:
        provider = create_provider(
            provider_name=config.provider,
            model=config.model,
            large_context_model=config.large_context_model,
            temperature=config.temperature,
            api_key='dummy-key-for-demo'
        )
        print(f"✓ Created {provider.get_provider_name()} provider")
        print(f"  Supports batch: {provider.supports_batch()}")
        print(f"  Supports caching: {provider.supports_caching()}")
    except Exception as e:
        print(f"⚠️  Google provider not available: {e}")
        print("   Install with: pip install google-generativeai")
    print()


def example_xai():
    """Example using xAI Grok."""
    print("=" * 70)
    print("XAI GROK EXAMPLE")
    print("=" * 70)

    config = FrameworkConfig(
        provider='xai',  # or 'grok'
        model='grok-4.1-fast',  # Or 'grok-4' for deeper reasoning
        k=1.5,
        r=0.3,
        temperature=1.0,
        use_batch_api=False  # xAI batch API not documented
    )

    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print("Note: Grok has automatic caching (>90% hit rates)")
    print("Note: Use 'grok-4' for deeper reasoning (slower)")
    print()

    try:
        provider = create_provider(
            provider_name=config.provider,
            model=config.model,
            temperature=config.temperature,
            api_key='dummy-key-for-demo'
        )
        print(f"✓ Created {provider.get_provider_name()} provider")
        print(f"  Supports batch: {provider.supports_batch()}")
        print(f"  Supports caching: {provider.supports_caching()}")
    except Exception as e:
        print(f"⚠️  xAI provider not available: {e}")
        print("   Install with: pip install openai  # xAI uses OpenAI SDK")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROVIDER COMPARISON EXAMPLE")
    print("=" * 70)
    print()

    show_available_providers()

    print("=" * 70)
    print("PROVIDER CONFIGURATION EXAMPLES")
    print("=" * 70)
    print()

    example_anthropic()
    example_openai()
    example_google()
    example_xai()

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Set your API key(s):")
    print("   export ANTHROPIC_API_KEY='your-key'")
    print("   export OPENAI_API_KEY='your-key'")
    print("   export GOOGLE_API_KEY='your-key'")
    print("   export XAI_API_KEY='your-key'")
    print()
    print("2. Install optional providers:")
    print("   pip install openai  # For OpenAI and xAI")
    print("   pip install google-generativeai  # For Google Gemini")
    print()
    print("3. Use FractalSummarizer with your chosen provider:")
    print("   from core import FractalSummarizer, FrameworkConfig")
    print("   config = FrameworkConfig(provider='anthropic', model='sonnet')")
    print("   summarizer = FractalSummarizer(config)")
    print()
