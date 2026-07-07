"""
Tests for Anthropic batch-usage accounting.

get_batch_results() must fold each succeeded request's usage into the
provider's cumulative total_usage — direct call() already does this, and
without the batch-side accounting, run_metadata.json underreports the
usage/cost of batch runs.
"""

from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from core.providers.anthropic_provider import AnthropicProvider


def _batch_result(custom_id, text, input_tokens, output_tokens,
                  cache_creation=0, cache_read=0):
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message),
    )


def _errored_result(custom_id):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(type="api_error"),
        ),
    )


def _provider_with_results(results):
    provider = AnthropicProvider(api_key="dummy-key-for-test")
    provider.client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(results=lambda batch_id: iter(results))
            )
        )
    )
    return provider


def test_get_batch_results_accumulates_total_usage():
    provider = _provider_with_results([
        _batch_result("req-0", "out0", input_tokens=100, output_tokens=40),
        _batch_result("req-1", "out1", input_tokens=250, output_tokens=60,
                      cache_creation=30, cache_read=20),
    ])

    outputs = provider.get_batch_results("batch-123")

    assert outputs == [("req-0", "out0", 40), ("req-1", "out1", 60)]

    usage = provider.get_total_usage()
    assert usage.input_tokens == 350
    assert usage.output_tokens == 100
    assert usage.cache_creation_tokens == 30
    assert usage.cache_read_tokens == 20


def test_get_batch_results_errored_requests_do_not_count():
    provider = _provider_with_results([
        _batch_result("req-0", "out0", input_tokens=10, output_tokens=5),
        _errored_result("req-1"),
    ])

    outputs = provider.get_batch_results("batch-456")

    assert outputs == [("req-0", "out0", 5)]

    usage = provider.get_total_usage()
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5


def test_get_batch_results_handles_missing_cache_fields():
    """Older/partial usage objects without cache fields must not crash."""
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="out")],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    result = SimpleNamespace(
        custom_id="req-0",
        result=SimpleNamespace(type="succeeded", message=message),
    )
    provider = _provider_with_results([result])

    outputs = provider.get_batch_results("batch-789")

    assert outputs == [("req-0", "out", 3)]
    usage = provider.get_total_usage()
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.cache_creation_tokens == 0
    assert usage.cache_read_tokens == 0
