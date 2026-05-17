"""
agent.py — BiasAgent with upgraded judgement and understanding.

Improvements over v2:
  1. Sensitive column tagging  — protected attributes flagged with higher scrutiny
  2. Chi-squared significance  — p-value filters out noise from small datasets
  3. Chain-of-thought synthesis— agent reasons across all columns after individual analysis
  4. Intersectional bias       — detects outcome gaps in column *combinations*
  5. Cramér's V compounding    — flags correlated biased columns that amplify each other
  6. Per-finding LLM explain   — one cheap call per biased column for plain-English insight

Total LLM calls:
  - 1 per biased column   (plain-English explanation — gpt-4o-mini, ~$0.0001 each)
  - 1 for the judge        (scoring — gpt-4o-mini)
  Typical total for Titanic: ~3–4 calls = ~$0.0003
"""

import json
import os
from itertools import combinations

import pandas as pd
from openai import OpenAI

from tools import (
    get_categorical_columns,
    get_distribution,
    detect_bias,
    imbalance_ratio,
    chi_squared_test,
    outcome_rate_by,
    outcome_gap,
    cramers_v,
    intersectional_bias,
    intersectional_outcome_gap,
    infer_column_context,
    missing_rate,
    dataset_summary,
)

MODEL = "gpt-4o-mini"
BIAS_THRESHOLD = 0.70
COMPOUNDING_THRESHOLD = 0.30  # Cramér's V above this = meaningfully correlated


class BiasAgent:
    """
    Autonomous bias detection agent for any CSV dataset.

    Parameters
    ----------
    df         : the uploaded DataFrame
    target_col : column for outcome-rate analysis (e.g. "Survived", "Churn")
    ignore_cols: column names to skip
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        ignore_cols: set | None = None,
    ):
        self.df = df
        self.target_col = target_col
        self.ignore_cols = ignore_cols or set()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        full_ignore = self.ignore_cols.copy()
        if target_col:
            full_ignore.add(target_col)

        self.columns = get_categorical_columns(df, ignore=full_ignore)
        self.log_lines: list[str] = []

    def _log(self, text: str):
        self.log_lines.append(text)

    # ── PHASE 1: Per-column analysis ─────────────────────────────────────────

    def run_detection_loop(self) -> tuple[str, list[dict]]:
        """
        Full autonomous pipeline. Returns (log_text, findings).

        Phase 1 — Per-column analysis (pure Python)
        Phase 2 — Chain-of-thought synthesis (pure Python)
        Phase 3 — Intersectional analysis on top biased pairs (pure Python)
        Phase 4 — Per-finding LLM explanations (1 LLM call per biased column)
        """
        self._log("=" * 65)
        self._log("AutoBiasAgent — Upgraded Detection Pipeline")
        self._log("=" * 65)

        summary = dataset_summary(self.df)
        self._log(f"Dataset          : {summary['rows']} rows × {summary['columns']} columns")
        self._log(f"Target column    : {self.target_col or '(none)'}")
        self._log(f"Columns to check : {self.columns}")
        if summary["missing_pct"]:
            self._log(f"Columns w/ NaNs  : {summary['missing_pct']}")
        self._log("")

        findings: list[dict] = []

        # ── Phase 1: individual column analysis ──────────────────────────────
        self._log("── PHASE 1: Individual Column Analysis ──────────────────────")
        for i, col in enumerate(self.columns):
            self._log(f"\n[step {i+1}] 🔧 TOOL CALLS → '{col}'")

            context   = infer_column_context(col)
            dist      = get_distribution(self.df, col)
            biased    = detect_bias(dist, BIAS_THRESHOLD)
            ratio     = imbalance_ratio(dist)
            chi2      = chi_squared_test(self.df, col, self.target_col) if self.target_col else {"p_value": None, "significant": None, "note": "No target."}
            outcome   = outcome_rate_by(self.df, col, self.target_col) if self.target_col else None
            gap       = outcome_gap(outcome)
            missing   = missing_rate(self.df, col)

            dominant_share = max(dist.values()) if dist else 0
            severity = dominant_share - BIAS_THRESHOLD if biased else 0.0

            # Risk level: combine bias flag + significance + outcome gap + sensitivity
            risk = _compute_risk(biased, chi2.get("significant"), gap, context["sensitive"])

            finding = {
                "column": col,
                "sensitive": context["sensitive"],
                "sensitive_reason": context["reason"],
                "distribution": dist,
                "biased": biased,
                "imbalance_ratio": ratio,
                "chi_squared": chi2,
                "outcome_rate": outcome,
                "outcome_gap": gap,
                "missing_rate": missing,
                "dominant_share": round(dominant_share, 4),
                "severity": round(severity, 4),
                "risk_level": risk,
                "explanation": None,       # filled in Phase 4
                "intersectional": None,    # filled in Phase 3
            }
            findings.append(finding)

            # Log
            if context["sensitive"]:
                self._log(f"         🔴 SENSITIVE ATTRIBUTE — {context['reason']}")
            pct_dist = {k: f"{v*100:.1f}%" for k, v in dist.items()}
            self._log(f"         Distribution   : {pct_dist}")
            self._log(f"         Imbalance ratio: {ratio}x")
            if missing > 0:
                self._log(f"         Missing values : {missing*100:.1f}%")
            if biased:
                dominant = max(dist, key=dist.get)
                self._log(f"         ⚠ BIAS DETECTED — '{dominant}' = {dominant_share*100:.1f}% (>{BIAS_THRESHOLD*100:.0f}%)")
            else:
                self._log(f"         ✓ Balanced — max {dominant_share*100:.1f}%")
            if chi2["p_value"] is not None:
                sig_icon = "📊 SIGNIFICANT" if chi2["significant"] else "〰 not significant"
                self._log(f"         Chi-squared    : {sig_icon} — {chi2['note']}")
            if outcome:
                out_pct = {k: f"{v*100:.1f}%" for k, v in outcome.items()}
                self._log(f"         {self.target_col} rate : {out_pct}  (gap: {gap*100:.1f}pp)")
            self._log(f"         Risk level     : {risk}")

        # ── Phase 2: Chain-of-thought synthesis ──────────────────────────────
        self._log("\n── PHASE 2: Chain-of-Thought Synthesis ──────────────────────")
        findings = self._synthesize(findings)

        # ── Phase 3: Intersectional analysis on top biased pairs ─────────────
        self._log("\n── PHASE 3: Intersectional Bias Analysis ────────────────────")
        findings = self._intersectional_pass(findings)

        # ── Phase 4: Per-finding LLM explanations ────────────────────────────
        self._log("\n── PHASE 4: LLM Plain-English Explanations ──────────────────")
        findings = self._explain_findings(findings)

        # ── Final ranking ─────────────────────────────────────────────────────
        findings.sort(key=lambda f: (_risk_order(f["risk_level"]), f["severity"]), reverse=True)

        self._log("\n🤖 FINAL RANKING (highest risk first):")
        for rank, f in enumerate(findings, 1):
            flag = "⚠" if f["biased"] else "✓"
            self._log(f"   {rank}. {f['column']:15} {flag}  risk={f['risk_level']:8}  ratio={f['imbalance_ratio']}x")

        biased_count = sum(1 for f in findings if f["biased"])
        self._log("\n" + "=" * 65)
        self._log(f"Columns analysed : {len(findings)}")
        self._log(f"Biased columns   : {biased_count}")
        self._log(f"Balanced columns : {len(findings) - biased_count}")
        self._log("=" * 65)

        return "\n".join(self.log_lines), findings

    # ── Phase 2: Synthesis ────────────────────────────────────────────────────

    def _synthesize(self, findings: list[dict]) -> list[dict]:
        """
        Chain-of-thought reasoning pass across all columns:
        - Compares columns that are both biased to check for Cramér's V compounding
        - Upgrades risk level when biased columns are also correlated
        """
        biased_cols = [f["column"] for f in findings if f["biased"]]

        if len(biased_cols) < 2:
            self._log("   Only one biased column — no compounding possible.")
            return findings

        self._log(f"   Checking compounding bias between {len(biased_cols)} biased columns...")

        compounding_pairs = []
        for col_a, col_b in combinations(biased_cols, 2):
            v = cramers_v(self.df, col_a, col_b)
            if v >= COMPOUNDING_THRESHOLD:
                compounding_pairs.append((col_a, col_b, v))
                self._log(
                    f"   ⚡ COMPOUNDING: '{col_a}' ↔ '{col_b}'  "
                    f"Cramér's V={v} — these biases amplify each other!"
                )

        # Upgrade risk for columns involved in compounding pairs
        compounding_cols = {col for pair in compounding_pairs for col in pair[:2]}
        for f in findings:
            if f["column"] in compounding_cols:
                if f["risk_level"] != "CRITICAL":
                    old = f["risk_level"]
                    f["risk_level"] = _upgrade_risk(f["risk_level"])
                    self._log(f"   ↑ '{f['column']}' risk upgraded: {old} → {f['risk_level']} (compounding)")

        if not compounding_pairs:
            self._log("   No significant compounding correlations found.")

        return findings

    # ── Phase 3: Intersectional analysis ─────────────────────────────────────

    def _intersectional_pass(self, findings: list[dict]) -> list[dict]:
        """
        For each pair of biased columns, compute intersectional outcome rates.
        Only runs if a target column is set.
        """
        if not self.target_col:
            self._log("   Skipped — no target column selected.")
            return findings

        biased = [f for f in findings if f["biased"]]
        if len(biased) < 2:
            self._log("   Fewer than 2 biased columns — no intersectional pairs.")
            return findings

        col_map = {f["column"]: f for f in findings}

        for col_a, col_b in combinations([f["column"] for f in biased], 2):
            self._log(f"   🔧 TOOL CALL → intersectional_bias('{col_a}', '{col_b}', '{self.target_col}')")
            inter = intersectional_bias(self.df, col_a, col_b, self.target_col)
            inter_gap = intersectional_outcome_gap(inter)

            if inter:
                best  = max(inter, key=inter.get)
                worst = min(inter, key=inter.get)
                self._log(f"      Best subgroup : {best} → {inter[best]*100:.1f}%")
                self._log(f"      Worst subgroup: {worst} → {inter[worst]*100:.1f}%")
                self._log(f"      Intersectional gap: {inter_gap*100:.1f}pp")

                # Attach to the first column in the pair
                if col_map[col_a]["intersectional"] is None:
                    col_map[col_a]["intersectional"] = {}
                col_map[col_a]["intersectional"][f"× {col_b}"] = {
                    "rates": inter,
                    "gap": inter_gap,
                }
            else:
                self._log(f"      No qualifying subgroups (groups too small).")

        return findings

    # ── Phase 4: Per-finding LLM explanations ────────────────────────────────

    def _explain_findings(self, findings: list[dict]) -> list[dict]:
        """
        One gpt-4o-mini call per biased column.
        Generates a plain-English 3-sentence explanation:
          1. What the bias is
          2. What the practical risk is
          3. One concrete fix
        """
        biased = [f for f in findings if f["biased"]]
        self._log(f"   Generating explanations for {len(biased)} biased column(s)...")

        for f in biased:
            col = f["column"]
            self._log(f"   🤖 LLM CALL → explain '{col}'")

            inter_summary = ""
            if f.get("intersectional"):
                for pair_key, pair_data in f["intersectional"].items():
                    inter_summary += (
                        f"\n  Intersectional analysis {pair_key}: "
                        f"subgroup rates = {pair_data['rates']}, "
                        f"gap = {pair_data['gap']*100:.1f}pp."
                    )

            prompt = f"""You are a data scientist explaining a bias finding to a non-technical stakeholder.

Column: '{col}'
Sensitive attribute: {f['sensitive']} — {f['sensitive_reason']}
Distribution: {f['distribution']}
Imbalance ratio: {f['imbalance_ratio']}x
Statistical significance: {f['chi_squared'].get('note', 'N/A')}
Outcome ({self.target_col}) rates: {f['outcome_rate']}
Outcome gap: {f['outcome_gap']*100:.1f} percentage points
Risk level: {f['risk_level']}{inter_summary}

Write EXACTLY 3 sentences:
1. What this bias is (plain English, include the key numbers).
2. What the practical risk is if a model trains on this data.
3. One concrete, actionable fix for this specific column.

Be specific. No preamble. No bullet points. Just 3 sentences."""

            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    temperature=0.2,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                explanation = response.choices[0].message.content.strip()
                f["explanation"] = explanation
                self._log(f"      → {explanation[:120]}...")
            except Exception as e:
                f["explanation"] = f"Explanation unavailable: {e}"
                self._log(f"      ⚠ LLM call failed: {e}")

        return findings

    # ── Final judge evaluation ────────────────────────────────────────────────

    def evaluate(self, findings: list[dict]) -> str:
        """
        ONE final OpenAI call — independent LLM judge scoring the full pipeline.
        Includes intersectional and compounding findings in the judge's context.
        """
        # Build a clean, judge-friendly summary
        clean = []
        for f in findings:
            entry = {
                "column": f["column"],
                "sensitive": f["sensitive"],
                "biased": f["biased"],
                "imbalance_ratio": f["imbalance_ratio"],
                "risk_level": f["risk_level"],
                "chi_squared_significant": f["chi_squared"].get("significant"),
                "outcome_gap_pct": round(f["outcome_gap"] * 100, 1) if f.get("outcome_gap") else None,
                "intersectional_pairs": list(f["intersectional"].keys()) if f.get("intersectional") else [],
                "explanation": f.get("explanation"),
            }
            clean.append(entry)

        dataset_ctx = (
            f"Dataset: {len(self.df)} rows, {len(self.df.columns)} columns. "
            f"Columns analysed: {[f['column'] for f in findings]}. "
            f"Target column: {self.target_col or 'none'}. "
            f"Sensitive columns found: {[f['column'] for f in findings if f['sensitive']]}."
        )

        prompt = (
            "You are a strict, independent evaluator of an AI bias-detection agent.\n\n"
            f"Context: {dataset_ctx}\n\n"
            "The agent ran a 4-phase pipeline:\n"
            "  Phase 1: Per-column distributional analysis + chi-squared significance tests\n"
            "  Phase 2: Chain-of-thought synthesis with Cramér's V compounding detection\n"
            "  Phase 3: Intersectional bias analysis across column pairs\n"
            "  Phase 4: Plain-English LLM explanations per biased column\n\n"
            "Findings:\n"
            f"{json.dumps(clean, indent=2)}\n\n"
            "Evaluate on these five criteria (score 1–5 each):\n"
            "1. Correctness    — Are the statistical findings accurate and well-supported?\n"
            "2. Coverage       — Were all meaningful columns and interactions checked?\n"
            "3. Insight        — Does the analysis surface real, non-obvious bias patterns?\n"
            "4. Reasoning      — Is the risk ranking and chain-of-thought logic sound?\n"
            "5. Actionability  — Would a data scientist find the explanations useful?\n\n"
            "Respond in EXACTLY this format:\n"
            "SCORES\n"
            "  Correctness   : X/5 — <one sentence>\n"
            "  Coverage      : X/5 — <one sentence>\n"
            "  Insight       : X/5 — <one sentence>\n"
            "  Reasoning     : X/5 — <one sentence>\n"
            "  Actionability : X/5 — <one sentence>\n\n"
            "OVERALL: X/25\n\n"
            "VERDICT\n<2–4 sentence qualitative summary>\n\n"
            "RECOMMENDATIONS\n<2–3 concrete bullet points for further improvement>"
        )

        response = self.client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()


# ── Risk helpers ──────────────────────────────────────────────────────────────

def _compute_risk(biased: bool, significant: bool | None, gap: float, sensitive: bool) -> str:
    """
    Assign a risk level based on four factors:
      CRITICAL — sensitive attribute + biased + significant + large outcome gap
      HIGH     — biased + (significant or large gap or sensitive)
      MEDIUM   — biased but not significant / small gap
      LOW      — not biased
    """
    if not biased:
        return "LOW"
    if sensitive and significant and gap >= 0.20:
        return "CRITICAL"
    if (significant and gap >= 0.15) or (sensitive and biased):
        return "HIGH"
    if biased:
        return "MEDIUM"
    return "LOW"


def _upgrade_risk(current: str) -> str:
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    idx = order.index(current) if current in order else 0
    return order[min(idx + 1, len(order) - 1)]


def _risk_order(risk: str) -> int:
    return {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}.get(risk, 0)