"""
Tests for the synthesis phase of BiasAgent.

No LLM calls happen here: `_synthesize` is pure statistics. A dummy API key is
set only so the constructor can build its OpenAI client, which is never used.

These exist because of a real crash. `_explain_findings` interpolated a variable
named `inter_summary` that was never defined, so every run with a working API
key died with NameError before reaching the try/except around the LLM call. The
compounding pairs it was meant to describe were computed, logged, and then
thrown away instead of being attached to the findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent import BiasAgent


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture
def correlated_df():
    """Sex and Title are heavily imbalanced and almost perfectly correlated."""
    n = 200
    sex = ["male"] * 170 + ["female"] * 30
    return pd.DataFrame(
        {
            "Sex": sex,
            "Title": ["Mr" if s == "male" else "Mrs" for s in sex],
            "Survived": [0] * 150 + [1] * 20 + [0] * 5 + [1] * 25,
        }
    )


def test_identifier_and_free_text_columns_are_skipped():
    """
    DEFAULT_IGNORE must reach get_categorical_columns. It previously did not:
    the agent always passed a set, and the defaults were only applied when the
    argument was None, so Name, Ticket and Cabin were analysed as categorical
    features on the bundled Titanic data.
    """
    df = pd.DataFrame(
        {
            "Name": [f"Passenger {i}" for i in range(60)],
            "Ticket": [f"T-{i}" for i in range(60)],
            "Cabin": [f"C{i}" for i in range(60)],
            "PassengerId": range(60),
            "Sex": ["male"] * 50 + ["female"] * 10,
            "Survived": [0] * 40 + [1] * 20,
        }
    )
    agent = BiasAgent(df, target_col="Survived")

    for skipped in ("Name", "Ticket", "Cabin", "PassengerId"):
        assert skipped not in agent.columns, f"{skipped} should be ignored by default"
    assert "Sex" in agent.columns


def test_caller_ignores_extend_rather_than_replace_the_defaults():
    df = pd.DataFrame(
        {
            "Name": [f"P{i}" for i in range(40)],
            "Sex": ["male"] * 30 + ["female"] * 10,
            "Deck": ["A"] * 20 + ["B"] * 20,
        }
    )
    agent = BiasAgent(df, ignore_cols={"Deck"})

    assert "Deck" not in agent.columns, "caller-supplied ignores must apply"
    assert "Name" not in agent.columns, "defaults must still apply alongside them"
    assert "Sex" in agent.columns


def test_compounding_columns_are_recorded_on_the_findings(correlated_df):
    agent = BiasAgent(correlated_df, target_col="Survived")
    _, findings = agent.run_detection_loop()

    by_col = {f["column"]: f for f in findings}
    assert by_col["Sex"]["compounding_with"], "Sex compounds with Title but was not recorded"
    partners = [other for other, _ in by_col["Sex"]["compounding_with"]]
    assert "Title" in partners


def test_compounding_is_symmetric(correlated_df):
    agent = BiasAgent(correlated_df, target_col="Survived")
    _, findings = agent.run_detection_loop()

    by_col = {f["column"]: f for f in findings}
    assert "Title" in [o for o, _ in by_col["Sex"]["compounding_with"]]
    assert "Sex" in [o for o, _ in by_col["Title"]["compounding_with"]]


def test_every_finding_has_the_field_even_without_compounding():
    """
    The early-return path (fewer than two biased columns) must still populate
    the field, or downstream consumers hit a KeyError.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Sex": ["male"] * 190 + ["female"] * 10,
            "Balanced": rng.choice(["a", "b"], 200),
            "Survived": rng.integers(0, 2, 200),
        }
    )
    agent = BiasAgent(df, target_col="Survived")
    _, findings = agent.run_detection_loop()

    for f in findings:
        assert "compounding_with" in f


def test_explanation_prompt_builds_without_a_live_api_call(correlated_df, monkeypatch):
    """
    Guards the original NameError: the prompt is an f-string built before the
    try/except, so an undefined name there takes down the whole run rather than
    degrading to "explanation unavailable".
    """
    agent = BiasAgent(correlated_df, target_col="Survived")

    captured = {}

    def fake_create(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(agent.client.chat.completions, "create", fake_create)

    _, findings = agent.run_detection_loop()

    assert "prompt" in captured, "the prompt was never built"
    assert "Compounding:" in captured["prompt"]
    assert "Title" in captured["prompt"] or "Sex" in captured["prompt"]
    # The LLM failure must be absorbed, not raised.
    assert any("unavailable" in (f.get("explanation") or "") for f in findings)
