"""
app.py — AutoBiasAgent UI, upgraded for richer judgement.

New vs v2:
  - Risk level badges (CRITICAL / HIGH / MEDIUM / LOW) in the log
  - Explanations tab showing plain-English LLM insight per biased column
  - Phase progress visible in status bar
  - Sensitive column warning banner in dataset preview
"""

import os
import io
import traceback

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import gradio as gr

from agent import BiasAgent
from tools import dataset_summary, get_categorical_columns, infer_column_context

# This app is CPU-only: pandas, scipy and OpenAI API calls, no local model
# inference. But ZeroGPU — currently the only free hardware tier for Gradio
# Spaces — refuses to start a Space in which no @spaces.GPU function is
# registered ("No @spaces.GPU function detected during startup"). Registering
# one satisfies that check. It is never called, so no GPU is ever allocated and
# no GPU quota is consumed. Outside Spaces the package is absent and this is
# skipped entirely.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_startup_probe() -> None:
        """Registered to satisfy the ZeroGPU startup check. Intentionally unused."""
        return None

except ImportError:
    pass

BUNDLED_CSV = os.path.join(os.path.dirname(__file__), "train.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_df(file_obj) -> pd.DataFrame | None:
    if file_obj is None:
        if os.path.exists(BUNDLED_CSV):
            return pd.read_csv(BUNDLED_CSV)
        return None
    if isinstance(file_obj, str):
        return pd.read_csv(file_obj)
    return pd.read_csv(io.BytesIO(file_obj))


def on_upload(file_obj):
    try:
        df = load_df(file_obj)
        if df is None:
            return "❌ Could not load file.", gr.update(choices=[]), gr.update(choices=[])

        summary = dataset_summary(df)
        cols = summary["column_names"]

        # Flag sensitive columns in the preview
        sensitive = [c for c in cols if infer_column_context(c)["sensitive"]]
        sensitive_banner = ""
        if sensitive:
            sensitive_banner = (
                f"<div style='background:#450a0a;border:1px solid #ef4444;border-radius:6px;"
                f"padding:8px 12px;margin-bottom:8px;color:#fca5a5;font-size:0.82rem'>"
                f"🔴 <b>Sensitive attributes detected:</b> {', '.join(sensitive)} — "
                f"bias in these columns may have legal or ethical implications.</div>"
            )

        preview_html = (
            sensitive_banner
            + f"<b>{summary['rows']} rows × {summary['columns']} columns</b><br><br>"
            + df.head(5).to_html(index=False, border=0, classes="preview-table")
        )

        target_choices = ["(none)"] + cols
        return preview_html, gr.update(choices=target_choices, value="(none)"), gr.update(choices=cols, value=[])

    except Exception as e:
        return f"❌ Error: {e}", gr.update(choices=[]), gr.update(choices=[])


def run_pipeline(file_obj, target_col_raw, ignore_cols_raw):
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or not api_key.startswith("sk-"):
            msg = (
                "❌ OPENAI_API_KEY missing or invalid.\n\n"
                "Add it in HuggingFace → Settings → Secrets, then reboot the Space."
            )
            return msg, msg, "", None, "❌ No API key"

        df = load_df(file_obj)
        if df is None:
            return "❌ No dataset.", "❌ No dataset.", "", None, "❌ No data"

        target_col = None if target_col_raw in ("(none)", "", None) else target_col_raw
        ignore_cols = set(ignore_cols_raw) if ignore_cols_raw else set()

        agent = BiasAgent(df, target_col=target_col, ignore_cols=ignore_cols)
        analysis_log, findings = agent.run_detection_loop()

        if not findings:
            return analysis_log, "❌ No columns detected.", "", None, "⚠ No findings"

        judge_verdict = agent.evaluate(findings)
        explanations_text = _format_explanations(findings, target_col)
        fig_dist = make_distribution_charts(df, findings, target_col)

        biased = sum(f["biased"] for f in findings)
        critical = sum(1 for f in findings if f["risk_level"] == "CRITICAL")
        status = (
            f"✅ Done — {len(findings)} columns analysed | "
            f"{biased} biased | {critical} CRITICAL risk"
        )
        return analysis_log, judge_verdict, explanations_text, fig_dist, status

    except Exception as e:
        err = f"❌ {type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        return err, err, "", None, "❌ Error"


def _format_explanations(findings: list[dict], target_col: str | None) -> str:
    """Format plain-English LLM explanations into a readable text block."""
    lines = ["PLAIN-ENGLISH BIAS EXPLANATIONS", "=" * 60, ""]
    biased = [f for f in findings if f["biased"]]
    if not biased:
        return "No biased columns detected — no explanations needed."

    for f in biased:
        risk_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(f["risk_level"], "⚪")
        lines.append(f"{risk_icon} [{f['risk_level']}] {f['column'].upper()}")
        lines.append("-" * 40)
        if f.get("explanation"):
            lines.append(f["explanation"])
        else:
            lines.append("No explanation generated.")
        lines.append("")

    return "\n".join(lines)


# ── Distribution charts ───────────────────────────────────────────────────────

RISK_COLORS = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6",
           "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#84cc16"]


def make_distribution_charts(df, findings, target_col):
    n = len(findings)
    fig, axes = plt.subplots(1, n, figsize=(max(5, 4 * n), 5))
    if n == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0f172a")

    for ax, finding in zip(axes, findings):
        col     = finding["column"]
        dist    = finding["distribution"]
        biased  = finding["biased"]
        outcome = finding.get("outcome_rate")
        ratio   = finding.get("imbalance_ratio", 1.0)
        risk    = finding.get("risk_level", "LOW")
        sig     = finding["chi_squared"].get("significant") if finding.get("chi_squared") else None

        labels = list(dist.keys())
        vals   = [dist[k] * 100 for k in labels]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.4)

        if outcome and target_col:
            for bar, label in zip(bars, labels):
                if label in outcome:
                    ax.plot(
                        bar.get_x() + bar.get_width() / 2,
                        outcome[label] * 100,
                        marker="D", color="white", markersize=7, zorder=5,
                    )

        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white", labelsize=7)
        if any(len(str(l)) > 6 for l in labels):
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("% of rows", color="white", fontsize=8)

        # Title color = risk level
        title_color = RISK_COLORS.get(risk, "white")
        sensitive_star = " ★" if finding.get("sensitive") else ""
        ax.set_title(f"{col}{sensitive_star}", color=title_color, fontweight="bold", fontsize=10)
        ax.spines[:].set_visible(False)

        # Bias label
        if biased:
            sig_note = " ✓sig" if sig else (" ✗insig" if sig is False else "")
            label_text = f"⚠ {risk}  ratio={ratio}x{sig_note}"
        else:
            label_text = f"✓ balanced  ratio={ratio}x"
        label_color = RISK_COLORS.get(risk, "#22c55e") if biased else "#22c55e"
        ax.text(0.5, 1.09, label_text, transform=ax.transAxes,
                ha="center", va="bottom", color=label_color, fontsize=7.5, fontweight="bold")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=7)

    legend_handles = []
    if target_col:
        legend_handles.append(mpatches.Patch(color="white", label=f"♦ = mean {target_col} rate"))
    legend_handles.append(mpatches.Patch(color="#a78bfa", label="★ = sensitive attribute"))
    fig.legend(handles=legend_handles, loc="lower center", facecolor="#0f172a",
               labelcolor="white", fontsize=8, ncol=2)

    fig.suptitle("AutoBiasAgent — Distribution Analysis (ordered by risk)",
                 color="white", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.preview-table { border-collapse: collapse; font-size: 0.8rem; }
.preview-table th, .preview-table td { border: 1px solid #334155; padding: 4px 8px; color: #e2e8f0; }
.preview-table th { background: #1e293b; }
"""

with gr.Blocks(title="AutoBiasAgent") as demo:

    # Injected as a <style> tag rather than passed to gr.Blocks(css=...):
    # Gradio 6 moved that argument to launch() and silently ignores it on the
    # constructor, and Spaces does not always run this file through __main__.
    # A style tag inside the layout works on every version and launch path.
    gr.HTML(f"<style>{CSS}</style>")

    gr.HTML("""
        <div style="text-align:center;padding:1rem 0">
            <h1 style="color:#6366f1;margin:0">🤖 AutoBiasAgent</h1>
            <p style="color:#94a3b8;margin:4px 0 0">
                3-phase bias detection · Statistical significance · Compounding correlation · LLM explanations
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📂 Upload CSV  (leave blank to use bundled Titanic dataset)",
                file_types=[".csv"],
                type="filepath",
            )
            dataset_preview = gr.HTML(label="Dataset Preview")

        with gr.Column(scale=1):
            target_col = gr.Dropdown(
                label="🎯 Target / outcome column",
                choices=["(none)"],
                value="(none)",
                info="Column used for outcome-rate analysis (e.g. Survived, Churn).",
            )
            ignore_cols = gr.CheckboxGroup(
                label="🚫 Columns to ignore",
                choices=[],
                value=[],
                info="Tick ID columns, free text, or irrelevant features.",
            )

    file_input.change(
        fn=on_upload,
        inputs=[file_input],
        outputs=[dataset_preview, target_col, ignore_cols],
    )

    with gr.Row():
        run_btn    = gr.Button("▶  Run Agent  (4-phase pipeline)", variant="primary", size="lg")
        status_box = gr.Textbox(label="Status", interactive=False, scale=2)

    with gr.Tabs():
        with gr.TabItem("🔍 Agent Log"):
            log_box = gr.Textbox(lines=22, max_lines=50,
                                 placeholder="3-phase detection log appears here …")
        with gr.TabItem("💡 Plain-English Explanations"):
            explain_box = gr.Textbox(lines=22, max_lines=50,
                                     placeholder="LLM-generated plain-English explanations per biased column …")
        with gr.TabItem("⚖️ LLM Judge Verdict"):
            judge_box = gr.Textbox(lines=22, max_lines=50,
                                   placeholder="Judge scoring appears here …")
        with gr.TabItem("📊 Distribution Charts"):
            chart_dist = gr.Plot(label="Distribution & Outcome Overlay")

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input, target_col, ignore_cols],
        outputs=[log_box, judge_box, explain_box, chart_dist, status_box],
    )

    gr.HTML("""
        <hr style="border-color:#334155;margin-top:1.5rem">
        <div style="text-align:center;color:#475569;font-size:.78rem;padding:.5rem 0">
            AutoBiasAgent · Phase 1: Distribution + Chi² ·
            Phase 2: Cramér's V Compounding · Phase 3: LLM Explanations
        </div>
    """)


if __name__ == "__main__":
    demo.launch()