---
title: bias_detector
emoji: 🤖
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---

# AutoBiasAgent

Autonomous 4-phase dataset bias detection — works on **any CSV**.

## What It Does

| Phase | What Happens |
|---|---|
| **Phase 1** | Per-column distribution analysis + chi-squared significance test |
| **Phase 2** | Chain-of-thought synthesis + Cramér's V compounding detection |
| **Phase 3** | Intersectional bias across column pairs |
| **Phase 4** | Plain-English LLM explanation per biased column |
| **Judge** | Independent GPT-4o-mini evaluation of all findings |

## Features

- Upload **any CSV** — not just Titanic
- Pick any column as the **target/outcome** (Survived, Churn, Label, etc.)
- **Sensitive attribute detection** — flags sex, race, age, religion, etc. automatically
- **Statistical significance** — p-values filter out noise from small datasets
- **Risk levels** — CRITICAL / HIGH / MEDIUM / LOW based on 4 combined factors
- **Compounding bias** — Cramér's V catches correlated biased columns that amplify each other
- **Intersectional analysis** — outcome rates across column *combinations* (e.g. Sex × Pclass)
- **Plain-English explanations** — LLM explains what each bias means and how to fix it

## Setup

1. Add `OPENAI_API_KEY` in **Settings → Secrets**
2. Click **Factory reboot**
3. Upload a CSV (or use the bundled Titanic fallback)
4. Select your target column and any columns to ignore
5. Click **▶ Run Agent**

## File Structure

```
app.py            — Gradio UI + pipeline orchestration
agent.py          — BiasAgent: 4-phase detection loop + LLM judge
tools.py          — Stateless analysis functions (pure Python + scipy)
prompts.py        — Prompt templates
requirements.txt
train.csv         — Bundled Titanic fallback dataset (optional)
```

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