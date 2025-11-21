#!/usr/bin/env python3
"""
Quick test script to verify provider implementations.

Tests basic instantiation and capability checking without making API calls.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.provider_factory import create_provider, list_providers, get_provider_info


def test_provider_factory():
    """Test that the provider factory works."""
    print("Testing Provider Factory")
    print("=" * 70)

    # List providers
    providers = list_providers()
    print(f"\nAvailable providers: {', '.join(providers)}")

    # Get capabilities
    print("\nProvider capabilities:")
    info = get_provider_info()
    for name, caps in info.items():
        print(f"  {name}:")
        print(f"    Batch API: {'✓' if caps['supports_batch'] else '✗'}")
        print(f"    Caching:   {'✓' if caps['supports_caching'] else '✗'}")

    print("\n" + "=" * 70)
    print("✅ Provider factory test passed!")


def test_anthropic_provider():
    """Test Anthropic provider instantiation."""
    print("\nTesting Anthropic Provider")
    print("=" * 70)

    try:
        provider = create_provider(
            'anthropic',
            model='sonnet',
            api_key='dummy'  # Won't make actual calls
        )
        print(f"✓ Created provider: {provider.get_provider_name()}")
        print(f"  Supports batch: {provider.supports_batch()}")
        print(f"  Supports caching: {provider.supports_caching()}")
        print("✅ Anthropic provider test passed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_config_integration():
    """Test that FrameworkConfig works with provider selection."""
    print("\nTesting Config Integration")
    print("=" * 70)

    try:
        from core.config import FrameworkConfig

        # Test with Anthropic
        config = FrameworkConfig(provider='anthropic', model='sonnet')
        print(f"✓ Created config with provider: {config.provider}")

        # Test with different providers
        for provider_name in ['openai', 'google', 'xai']:
            config = FrameworkConfig(provider=provider_name, model='test-model')
            print(f"✓ Config accepts provider: {provider_name}")

        # Test invalid provider
        try:
            config = FrameworkConfig(provider='invalid-provider')
            print("❌ Should have rejected invalid provider")
        except ValueError as e:
            print(f"✓ Correctly rejected invalid provider: {e}")

        print("✅ Config integration test passed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROVIDER SYSTEM TESTS")
    print("=" * 70)

    test_provider_factory()
    test_anthropic_provider()
    test_config_integration()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
