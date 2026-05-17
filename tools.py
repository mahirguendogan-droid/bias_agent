"""
tools.py — Stateless analysis tools for AutoBiasAgent.

Upgrades in this version:
  - Sensitive column tagging (protected attributes get higher scrutiny)
  - Chi-squared significance testing (p-value per column)
  - Intersectional bias detection (outcome gaps across column pairs)
  - Cramér's V correlation between biased columns (compounding bias)
  - Imbalance ratio (continuous severity beyond binary flag)
  - Dynamic target column (not hardcoded to "Survived")
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# ── Default columns to skip ───────────────────────────────────────────────────
DEFAULT_IGNORE = {"Name", "Ticket", "Cabin", "PassengerId", "id", "ID", "index"}

# ── Sensitive / protected attributes ─────────────────────────────────────────
SENSITIVE_KEYWORDS = {
    "sex", "gender", "race", "ethnicity", "ethnic", "age", "religion",
    "nationality", "nation", "disability", "marital", "pregnant",
    "orientation", "colour", "color", "caste", "diagnosis", "salary"
}


def infer_column_context(col_name: str) -> dict:
    """
    Classify a column as sensitive (protected attribute) or standard.
    Returns dict with 'sensitive' (bool) and 'reason' (str).
    """
    lower = col_name.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in lower:
            return {
                "sensitive": True,
                "reason": (
                    f"'{col_name}' matches protected attribute '{keyword}'. "
                    "Bias here may have legal/ethical implications."
                ),
            }
    return {"sensitive": False, "reason": "Standard feature — no protected-attribute match."}


def get_categorical_columns(df: pd.DataFrame, ignore: set | None = None) -> list:
    """
    Return columns worth analysing:
      - object / category / string dtype
      - Low-cardinality integers (2–20 unique values)
    Sensitive columns are sorted to the front.
    """
    if ignore is None:
        ignore = DEFAULT_IGNORE

    standard, sensitive = [], []
    for col in df.columns:
        if col in ignore:
            continue
        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)

        is_text = (
            dtype == "object"
            or dtype.name in ("category", "string")
            or str(dtype) in ("object", "string", "str")
            or hasattr(dtype, "categories")
        )
        is_low_card_int = pd.api.types.is_integer_dtype(dtype) and 2 <= n_unique <= 20

        if is_text or is_low_card_int:
            if infer_column_context(col)["sensitive"]:
                sensitive.append(col)
            else:
                standard.append(col)

    return sensitive + standard


def get_distribution(df: pd.DataFrame, column: str) -> dict:
    """Proportion of each value in column (NaNs excluded)."""
    return df[column].dropna().value_counts(normalize=True).to_dict()


def detect_bias(distribution: dict, threshold: float = 0.70) -> bool:
    """True if any single category exceeds threshold share."""
    if not distribution:
        return False
    return max(distribution.values()) > threshold


def imbalance_ratio(distribution: dict) -> float:
    """Ratio of most to least common category. 1.0 = balanced."""
    if not distribution or len(distribution) < 2:
        return 1.0
    vals = list(distribution.values())
    return round(max(vals) / min(vals), 2) if min(vals) > 0 else float("inf")


def chi_squared_test(df: pd.DataFrame, col: str, target: str) -> dict:
    """
    Chi-squared test of independence between col and target.
    Returns p_value, significant (bool), and a human-readable note.
    """
    if target not in df.columns:
        return {"p_value": None, "significant": None, "note": "No target column provided."}
    try:
        ct = pd.crosstab(df[col].dropna(), df[target].dropna())
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {"p_value": None, "significant": None, "note": "Insufficient categories for test."}
        _, p, _, _ = chi2_contingency(ct)
        p = round(float(p), 5)
        significant = p < 0.05
        note = (
            f"p={p} — statistically significant (bias is real, not random chance)."
            if significant
            else f"p={p} — not statistically significant (may be random variation)."
        )
        return {"p_value": p, "significant": significant, "note": note}
    except Exception as e:
        return {"p_value": None, "significant": None, "note": f"Test failed: {e}"}


def outcome_rate_by(df: pd.DataFrame, column: str, target: str) -> dict | None:
    """Mean value of target for each category in column."""
    if target not in df.columns:
        return None
    try:
        return df.groupby(column)[target].mean().round(3).to_dict()
    except Exception:
        return None


def outcome_gap(outcome_rates: dict | None) -> float:
    """Max minus min outcome rate across groups. High gap = high disparity."""
    if not outcome_rates or len(outcome_rates) < 2:
        return 0.0
    vals = list(outcome_rates.values())
    return round(max(vals) - min(vals), 4)


def cramers_v(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """
    Cramér's V — association between two categorical columns.
    0.0 = independent, 1.0 = perfectly correlated.
    High value between two biased columns = compounding bias risk.
    """
    try:
        ct = pd.crosstab(df[col_a].dropna(), df[col_b].dropna())
        chi2, _, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        r, k = ct.shape
        v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
        return round(float(v), 3)
    except Exception:
        return 0.0


def intersectional_bias(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    target: str,
    min_group_size: int = 10,
) -> dict | None:
    """
    Outcome rate for every combination of col_a × col_b.
    Only includes subgroups with >= min_group_size rows (small cells are noisy).

    Example:
      ("female", "1st") → 0.968
      ("male",   "3rd") → 0.135
    A large spread across subgroups = intersectional bias.
    """
    if target not in df.columns:
        return None
    try:
        grouped = (
            df.dropna(subset=[col_a, col_b, target])
            .groupby([col_a, col_b])[target]
            .agg(["mean", "count"])
            .reset_index()
        )
        result = {}
        for _, row in grouped.iterrows():
            if row["count"] >= min_group_size:
                key = f"{row[col_a]} × {row[col_b]}"
                result[key] = round(float(row["mean"]), 3)
        return result if result else None
    except Exception:
        return None


def intersectional_outcome_gap(intersectional: dict | None) -> float:
    """Max minus min outcome across all intersectional subgroups."""
    if not intersectional or len(intersectional) < 2:
        return 0.0
    vals = list(intersectional.values())
    return round(max(vals) - min(vals), 4)


def missing_rate(df: pd.DataFrame, column: str) -> float:
    """Fraction of rows where column is NaN."""
    return round(df[column].isna().mean(), 4)


def dataset_summary(df: pd.DataFrame) -> dict:
    """High-level summary of the uploaded dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_pct": {
            col: round(df[col].isna().mean() * 100, 1)
            for col in df.columns
            if df[col].isna().any()
        },
    }