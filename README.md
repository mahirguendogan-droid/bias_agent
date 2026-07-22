---
title: AutoBiasAgent
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: "Bias detection for any CSV: chi2, Cramer's V, LLM judge"
---

# AutoBiasAgent

[![Open in Spaces](https://img.shields.io/badge/▶️_Try_it_live-Hugging_Face_Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mahirgundogan/AutoBiasAgent)

[![CI](https://github.com/mahirguendogan-droid/bias_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/mahirguendogan-droid/bias_agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)

Autonomous 3-phase dataset bias detection — works on **any CSV**.

Upload a dataset, pick the outcome column, and the agent decides on its own which columns
deserve scrutiny, whether the imbalance it finds is statistically real, whether two biases
compound each other, and what any of it means in plain English — for roughly **$0.0005 a run**.

See **[DEPLOY.md](DEPLOY.md)** to run it as a free Hugging Face Space.

## What It Does

| Phase | What Happens |
|---|---|
| **Phase 1** | Per-column distribution analysis + chi-squared significance test |
| **Phase 2** | Chain-of-thought synthesis + Cramér's V compounding detection |
| **Phase 3** | Plain-English LLM explanation per biased column |
| **Judge** | Independent GPT-4o-mini evaluation of all findings |

## Features

- Upload **any CSV** — not just Titanic
- Pick any column as the **target/outcome** (Survived, Churn, Label, etc.)
- **Sensitive attribute detection** — flags sex, race, age, religion, etc. automatically
- **Statistical significance** — p-values filter out noise from small datasets
- **Risk levels** — CRITICAL / HIGH / MEDIUM / LOW based on 4 combined factors
- **Compounding bias** — Cramér's V catches correlated biased columns that amplify each other
- **Plain-English explanations** — LLM explains what each bias means and how to fix it

## Setup

### On Hugging Face Spaces

1. Add `OPENAI_API_KEY` in **Settings → Variables and secrets**
2. Click **Factory reboot**
3. Upload a CSV (or use the bundled Titanic fallback)
4. Select your target column and any columns to ignore
5. Click **▶ Run Agent**

Full deployment walkthrough: **[DEPLOY.md](DEPLOY.md)**.

### Locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # Windows: $env:OPENAI_API_KEY="sk-..."
python app.py                        # serves on http://127.0.0.1:7860
```

Without a key the statistical phases still run in full; only the LLM explanation and judge
phases are unavailable.

## File Structure

```
app.py            — Gradio UI + pipeline orchestration
agent.py          — BiasAgent: 3-phase detection loop + LLM judge
tools.py          — Stateless analysis functions (pure Python + scipy)
prompts.py        — Prompt templates
tests/            — Unit tests for the statistical layer (no API key needed)
requirements.txt
train.csv         — Bundled Titanic fallback dataset (optional)
```

## Tests

The statistical layer is what every conclusion rests on, so it is covered offline — no API
key, no network:

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

These tests earned their keep immediately by catching two defects in `cramers_v`:

- **NaN escaped the function.** A constant column makes `min(r,k) - 1 == 0`; NumPy returns
  `nan` for that division instead of raising, so the `except` never fired and a `nan` leaked
  out of a function documented to return 0.0–1.0.
- **Perfect correlation scored 0.98, not 1.0.** `chi2_contingency` applies Yates' continuity
  correction to 2×2 tables by default, which deflates χ² and makes Cramér's V understate the
  association — precisely in the binary sensitive-attribute case this tool exists to examine.
  The correction is reasonable for a hypothesis test and wrong for an effect size, so it is
  now disabled for V.

## Risk Level Logic

| Level | Conditions |
|---|---|
| **CRITICAL** | Sensitive attribute + biased + statistically significant + large outcome gap (≥20pp) |
| **HIGH** | Biased + significant + gap ≥15pp OR sensitive attribute + biased |
| **MEDIUM** | Biased but not significant or small gap |
| **LOW** | Not biased |

Columns involved in compounding pairs (Cramér's V ≥ 0.30) are upgraded one risk level.

## Cost

~$0.0001 per LLM call (GPT-4o-mini).
Typical run: 3–5 calls = **~$0.0003–0.0005 total**.