#!/usr/bin/env python3
"""
Provider factory for creating LLM provider instances.

Simplifies provider instantiation based on configuration.
"""

from typing import Optional

from .base_provider import BaseProvider
from .providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    XAIProvider,
)


# Provider registry - only include available providers
PROVIDERS = {
    'anthropic': AnthropicProvider,
}

if OpenAIProvider is not None:
    PROVIDERS['openai'] = OpenAIProvider

if GoogleProvider is not None:
    PROVIDERS['google'] = GoogleProvider
    PROVIDERS['gemini'] = GoogleProvider  # Alias

if XAIProvider is not None:
    PROVIDERS['xai'] = XAIProvider
    PROVIDERS['grok'] = XAIProvider  # Alias


def create_provider(
    provider_name: str,
    model: str,
    large_context_model: Optional[str] = None,
    temperature: float = 1.0,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    Create a provider instance.

    Args:
        provider_name: Provider name ('anthropic', 'openai', 'google', 'xai')
        model: Model name for small contexts
        large_context_model: Model for large contexts (provider-specific)
        temperature: Sampling temperature
        api_key: API key (uses provider-specific env var if None)
        **kwargs: Provider-specific parameters

    Returns:
        Provider instance

    Raises:
        ValueError: If provider_name is not recognized

    Example:
        >>> provider = create_provider('anthropic', model='sonnet')
        >>> output, tokens, usage = provider.call("Hello", "small")
    """
    provider_name = provider_name.lower()

    if provider_name not in PROVIDERS:
        available = ', '.join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {available}"
        )

    provider_class = PROVIDERS[provider_name]

    return provider_class(
        model=model,
        large_context_model=large_context_model,
        temperature=temperature,
        api_key=api_key,
        **kwargs
    )


def list_providers() -> list:
    """
    List available provider names.

    Returns:
        List of provider names
    """
    # Return unique provider names (remove aliases)
    unique = set()
    for name, cls in PROVIDERS.items():
        if name in ['anthropic', 'openai', 'google', 'xai']:
            unique.add(name)
    return sorted(unique)


def get_provider_info() -> dict:
    """
    Get information about available providers.

    Returns:
        Dict mapping provider names to their capabilities
    """
    info = {}

    # Get unique provider names
    unique_names = set()
    for name, cls in PROVIDERS.items():
        if cls is not None:
            # Get the base name (without aliases)
            if name in ['anthropic', 'openai', 'google', 'xai']:
                unique_names.add(name)

    for name in unique_names:
        # Create a temporary instance to check capabilities
        # (we don't call API, just instantiate)
        try:
            # Use dummy API key for capability checking
            provider_class = PROVIDERS[name]
            if name == 'anthropic':
                provider = provider_class(model='sonnet', api_key='dummy')
            elif name == 'openai':
                provider = provider_class(model='gpt-5.1', api_key='dummy')
            elif name == 'google':
                provider = provider_class(model='gemini-2.5-flash', api_key='dummy')
            elif name == 'xai':
                provider = provider_class(model='grok-4.1-fast', api_key='dummy')

            info[name] = {
                'supports_batch': provider.supports_batch(),
                'supports_caching': provider.supports_caching(),
                'available': True,
            }
        except Exception:
            # If instantiation fails, just mark as unknown
            info[name] = {
                'supports_batch': False,
                'supports_caching': False,
                'available': False,
            }

    return info


if __name__ == "__main__":
    print("Available providers:")
    print()

    for provider, capabilities in get_provider_info().items():
        print(f"  {provider}:")
        print(f"    Batch API: {'✓' if capabilities['supports_batch'] else '✗'}")
        print(f"    Caching:   {'✓' if capabilities['supports_caching'] else '✗'}")
        print()

    print("\nExample usage:")
    print("  from core.provider_factory import create_provider")
    print("  provider = create_provider('anthropic', model='sonnet')")
    print("  output, tokens, usage = provider.call('Hello', 'small')")
