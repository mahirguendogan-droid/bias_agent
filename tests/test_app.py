"""
Tests for the UI callback contract.

run_pipeline feeds five Gradio components:
    [log_box, judge_box, explain_box, chart_dist, status_box]

Every return path must therefore produce exactly five values. The failure paths
used to return six — an extra None left over from an earlier layout — so any
error, including the common "no API key yet" case, surfaced as a raw Gradio
arity error instead of the message the code carefully prepared.

Nothing here calls OpenAI: each case is chosen to return before the LLM phases.
"""

from __future__ import annotations

import pandas as pd
import pytest

import app

EXPECTED_OUTPUTS = 5


@pytest.fixture
def valid_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")


def test_missing_api_key_returns_the_full_output_tuple(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = app.run_pipeline(None, "(none)", [])
    assert len(result) == EXPECTED_OUTPUTS
    assert "OPENAI_API_KEY" in result[0]


def test_malformed_api_key_is_rejected_like_a_missing_one(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    result = app.run_pipeline(None, "(none)", [])
    assert len(result) == EXPECTED_OUTPUTS
    assert result[-1] == "❌ No API key"


def test_unreadable_upload_returns_the_full_output_tuple(valid_key, monkeypatch):
    monkeypatch.setattr(app, "load_df", lambda _f: None)
    result = app.run_pipeline(None, "(none)", [])
    assert len(result) == EXPECTED_OUTPUTS
    assert result[-1] == "❌ No data"


def test_unexpected_exception_returns_the_full_output_tuple(valid_key, monkeypatch):
    def boom(_f):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(app, "load_df", boom)
    result = app.run_pipeline(None, "(none)", [])
    assert len(result) == EXPECTED_OUTPUTS
    assert "disk on fire" in result[0]
    assert result[-1] == "❌ Error"


def test_dataset_with_no_analysable_columns(valid_key, monkeypatch):
    """Continuous columns are not categorical, so nothing is left to analyse."""
    frame = pd.DataFrame({"Fare": [float(i) * 1.5 for i in range(40)]})
    monkeypatch.setattr(app, "load_df", lambda _f: frame)
    result = app.run_pipeline(None, "(none)", [])
    assert len(result) == EXPECTED_OUTPUTS
    assert result[-1] == "⚠ No findings"
