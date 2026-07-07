"""
Tests for final-synthesis model-tier selection and batch-flag honesty
in FractalSummarizer.

The convergence gate (target_convergence) may legally exceed the
small-context budget T1, so the final synthesis must choose its tier
from the actual size of the final prompt (and fail loudly if even the
large tier cannot hold it). Batch configuration must also reflect what
can actually run.
"""

import json

import pytest

from core.base_provider import UsageStats
from core.config import AnalysisConfig, FrameworkConfig
from core.document import Document
from core.fractal_summarizer import FractalSummarizer


class StubProvider:
    """Minimal provider stub that records the context size of each call."""

    def __init__(self, supports_batch=False):
        self.context_sizes = []
        self._supports_batch = supports_batch
        self.total_usage = UsageStats()

    def call(self, prompt, context_size, max_tokens=50_000, timeout=600.0):
        self.context_sizes.append(context_size)
        usage = UsageStats(input_tokens=len(prompt) // 4, output_tokens=10)
        self.total_usage = self.total_usage + usage
        return "stub synthesis output", usage.output_tokens, usage

    def supports_batch(self):
        return self._supports_batch

    def supports_caching(self):
        return False

    def get_total_usage(self):
        return self.total_usage

    def reset_usage(self):
        self.total_usage = UsageStats()

    def get_provider_name(self):
        return "stub"

    def calculate_cost(self, usage, model):
        return 0.0


def _make_summarizer(monkeypatch, provider=None, **config_kwargs):
    provider = provider or StubProvider()
    monkeypatch.setattr(
        "core.fractal_summarizer.create_provider",
        lambda **kwargs: provider,
    )
    config = FrameworkConfig(**config_kwargs)
    return FractalSummarizer(config), provider


def _analysis_config(tmp_path):
    return AnalysisConfig(
        name="tier test",
        layer_prompt_template="{documents}",
        final_synthesis_prompt="Synthesize the findings:",
        output_dir=str(tmp_path / "out"),
    )


def _doc(n_words, doc_id="doc"):
    content = ("word " * n_words).strip()
    # token_count only needs to be positive; the tier decision counts the
    # actual prompt with the tokenizer, not this field.
    return Document(content=content, token_count=max(1, n_words), doc_id=doc_id)


def test_final_synthesis_uses_small_tier_when_prompt_fits_t1(monkeypatch, tmp_path):
    summarizer, provider = _make_summarizer(
        monkeypatch, T1=500, T2=2_000, target_convergence=1_800
    )

    result = summarizer._run_final_synthesis([_doc(50)], _analysis_config(tmp_path))

    assert result == "stub synthesis output"
    assert provider.context_sizes == ["small"]


def test_final_synthesis_uses_large_tier_when_prompt_exceeds_t1(monkeypatch, tmp_path):
    summarizer, provider = _make_summarizer(
        monkeypatch, T1=200, T2=5_000, target_convergence=4_000
    )

    # ~1,000 tokens of prompt: over T1=200, under T2=5,000
    result = summarizer._run_final_synthesis([_doc(1_000)], _analysis_config(tmp_path))

    assert result == "stub synthesis output"
    assert provider.context_sizes == ["large"]


def test_final_synthesis_fails_loud_when_prompt_exceeds_t2(monkeypatch, tmp_path):
    summarizer, provider = _make_summarizer(
        monkeypatch, T1=200, T2=600, target_convergence=500
    )

    with pytest.raises(ValueError, match="exceeds the large-context budget"):
        summarizer._run_final_synthesis([_doc(2_000)], _analysis_config(tmp_path))

    # No API call should have been attempted
    assert provider.context_sizes == []


def test_use_batch_api_with_unsupporting_provider_fails_at_construction(monkeypatch):
    provider = StubProvider(supports_batch=False)
    monkeypatch.setattr(
        "core.fractal_summarizer.create_provider",
        lambda **kwargs: provider,
    )

    with pytest.raises(ValueError, match="does not support the batch API"):
        FractalSummarizer(FrameworkConfig(use_batch_api=True))


def test_run_metadata_records_actual_batch_mode(monkeypatch, tmp_path):
    """run_metadata.json must record the batch mode that actually ran."""
    summarizer, provider = _make_summarizer(
        monkeypatch, T1=500, T2=2_000, target_convergence=1_800
    )

    docs = [_doc(30, doc_id=f"doc_{i}") for i in range(3)]
    analysis_config = _analysis_config(tmp_path)

    _, metadata = summarizer.run(docs, analysis_config)

    assert metadata.use_batch_api is False

    with open(tmp_path / "out" / "run_metadata.json") as f:
        saved = json.load(f)
    assert saved["use_batch_api"] is False
