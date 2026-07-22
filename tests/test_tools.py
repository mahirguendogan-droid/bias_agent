"""
Unit tests for tools.py — the statistical layer of AutoBiasAgent.

These cover the pure analysis functions only: no OpenAI calls, no API key,
no network. That is deliberate — the statistics are what the agent's
conclusions rest on, so they are the part worth pinning down in CI.
"""

import numpy as np
import pandas as pd
import pytest

from tools import (
    DEFAULT_IGNORE,
    chi_squared_test,
    cramers_v,
    dataset_summary,
    detect_bias,
    get_categorical_columns,
    get_distribution,
    imbalance_ratio,
    infer_column_context,
    missing_rate,
    outcome_gap,
    outcome_rate_by,
)


@pytest.fixture
def df():
    """Small frame with a deliberate association between Sex and Survived."""
    return pd.DataFrame(
        {
            "Sex": ["male"] * 60 + ["female"] * 40,
            "Survived": [0] * 50 + [1] * 10 + [0] * 5 + [1] * 35,
            "Pclass": [1, 2, 3] * 33 + [1],
            "Fare": np.linspace(5.0, 500.0, 100),
            "PassengerId": range(100),
        }
    )


# ── infer_column_context ──────────────────────────────────────────────────────

@pytest.mark.parametrize("col", ["Sex", "gender", "RACE", "Age", "marital_status"])
def test_protected_attributes_are_flagged(col):
    assert infer_column_context(col)["sensitive"] is True


@pytest.mark.parametrize("col", ["Fare", "Pclass", "Embarked"])
def test_ordinary_columns_are_not_flagged(col):
    assert infer_column_context(col)["sensitive"] is False


def test_sensitivity_check_is_case_insensitive():
    assert infer_column_context("GENDER")["sensitive"] is True
    assert infer_column_context("gender")["sensitive"] is True


def test_context_always_explains_itself():
    for col in ("Sex", "Fare"):
        assert infer_column_context(col)["reason"].strip()


# ── get_categorical_columns ───────────────────────────────────────────────────

def test_sensitive_columns_are_analysed_first(df):
    cols = get_categorical_columns(df)
    assert cols[0] == "Sex", "protected attributes must be surfaced before ordinary ones"


def test_identifier_columns_are_skipped(df):
    assert "PassengerId" in DEFAULT_IGNORE
    assert "PassengerId" not in get_categorical_columns(df)


def test_low_cardinality_integers_are_included(df):
    assert "Pclass" in get_categorical_columns(df)


def test_continuous_columns_are_excluded(df):
    assert "Fare" not in get_categorical_columns(df)


def test_high_cardinality_integers_are_excluded():
    frame = pd.DataFrame({"code": range(100)})
    assert get_categorical_columns(frame) == []


# ── get_distribution / detect_bias / imbalance_ratio ──────────────────────────

def test_distribution_is_normalised(df):
    dist = get_distribution(df, "Sex")
    assert pytest.approx(sum(dist.values())) == 1.0
    assert pytest.approx(dist["male"]) == 0.6


def test_distribution_ignores_missing_values():
    frame = pd.DataFrame({"x": ["a", "a", "b", None]})
    dist = get_distribution(frame, "x")
    assert pytest.approx(dist["a"]) == 2 / 3


def test_bias_flag_respects_threshold():
    assert detect_bias({"a": 0.8, "b": 0.2}) is True
    assert detect_bias({"a": 0.6, "b": 0.4}) is False


def test_bias_flag_is_strict_at_the_boundary():
    """0.70 exactly must not trip a 'greater than 0.70' rule."""
    assert detect_bias({"a": 0.70, "b": 0.30}) is False


def test_empty_distribution_is_not_biased():
    assert detect_bias({}) is False


def test_imbalance_ratio_reports_balanced_and_skewed():
    assert imbalance_ratio({"a": 0.5, "b": 0.5}) == 1.0
    assert imbalance_ratio({"a": 0.9, "b": 0.1}) == 9.0


def test_single_category_has_no_imbalance():
    assert imbalance_ratio({"a": 1.0}) == 1.0


# ── chi_squared_test ──────────────────────────────────────────────────────────

def test_real_association_is_significant(df):
    result = chi_squared_test(df, "Sex", "Survived")
    assert result["significant"] is True
    assert result["p_value"] < 0.05


def test_independent_columns_are_not_significant():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"x": rng.choice(["a", "b"], 400), "y": rng.choice([0, 1], 400)}
    )
    assert chi_squared_test(frame, "x", "y")["significant"] is False


def test_missing_target_is_reported_not_raised(df):
    result = chi_squared_test(df, "Sex", "NoSuchColumn")
    assert result["p_value"] is None
    assert "no target" in result["note"].lower()


def test_single_category_cannot_be_tested():
    frame = pd.DataFrame({"x": ["a"] * 10, "y": [0, 1] * 5})
    assert chi_squared_test(frame, "x", "y")["p_value"] is None


# ── outcome_rate_by / outcome_gap ─────────────────────────────────────────────

def test_outcome_rates_are_per_group(df):
    rates = outcome_rate_by(df, "Sex", "Survived")
    assert rates["female"] > rates["male"]


def test_outcome_gap_is_the_spread(df):
    rates = outcome_rate_by(df, "Sex", "Survived")
    assert pytest.approx(outcome_gap(rates)) == round(
        max(rates.values()) - min(rates.values()), 4
    )


def test_outcome_gap_of_nothing_is_zero():
    assert outcome_gap(None) == 0.0
    assert outcome_gap({"only": 0.5}) == 0.0


# ── cramers_v ─────────────────────────────────────────────────────────────────

def test_identical_columns_are_perfectly_associated():
    frame = pd.DataFrame({"a": ["x", "y"] * 50})
    frame["b"] = frame["a"]
    assert cramers_v(frame, "a", "b") == pytest.approx(1.0)


def test_independent_columns_have_low_association():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"a": rng.choice(["x", "y"], 500), "b": rng.choice(["p", "q"], 500)}
    )
    assert cramers_v(frame, "a", "b") < 0.2


def test_constant_column_degrades_to_zero():
    frame = pd.DataFrame({"a": ["x"] * 20, "b": ["p", "q"] * 10})
    assert cramers_v(frame, "a", "b") == 0.0


# ── missing_rate / dataset_summary ────────────────────────────────────────────

def test_missing_rate_counts_nulls():
    frame = pd.DataFrame({"x": [1, 2, None, None]})
    assert missing_rate(frame, "x") == 0.5


def test_dataset_summary_reports_shape(df):
    summary = dataset_summary(df)
    assert summary["rows"] == 100
    assert summary["columns"] == len(df.columns)
    assert "Sex" in summary["column_names"]


def test_summary_only_lists_columns_that_are_missing_data(df):
    assert dataset_summary(df)["missing_pct"] == {}
