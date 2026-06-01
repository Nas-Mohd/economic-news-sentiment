"""
utils/plot.py
─────────────
Visualisation utilities:
  • plot_training_curves()  — loss & val F1 across phases for one model
  • plot_comparison()       — side-by-side per-aspect F1 bar chart
  • plot_threshold_curves() — F1 vs threshold per aspect

Run standalone after training:
    python utils/plot.py --output_dir outputs/
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ASPECT_COLS = [
    "Monetary_Financial",
    "Inflation_Prices",
    "Real_Economic_Activity",
    "Labor_Consumption",
    "Fiscal_Government",
    "External_Sector",
]

ASPECT_SHORT = {
    "Monetary_Financial":     "Monetary",
    "Inflation_Prices":       "Inflation",
    "Real_Economic_Activity": "RealEcon",
    "Labor_Consumption":      "Labor",
    "Fiscal_Government":      "Fiscal",
    "External_Sector":        "External",
}

COLORS = {
    "Baseline (FinBERT)":      "#5b8db8",
    "Modern-FinBERT-large":    "#e07b39",
}


# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_path: str, model_label: str, save_dir: str):
    with open(log_path) as f:
        log = json.load(f)

    df = pd.DataFrame(log)
    phases = sorted(df["phase"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training curves — {model_label}", fontsize=13)

    phase_colors = {1: "#4e9af1", 2: "#e07b39", 3: "#6bc96b"}
    offset = 0

    for phase in phases:
        ph = df[df["phase"] == phase].reset_index(drop=True)
        x  = np.arange(offset, offset + len(ph)) + 1
        c  = phase_colors[phase]

        axes[0].plot(x, ph["train_loss"], color=c, linewidth=2, label=f"Phase {phase} train")
        axes[0].plot(x, ph["val_loss"],   color=c, linewidth=2, linestyle="--")
        axes[1].plot(x, ph["val_macro_f1"], color=c, linewidth=2, label=f"Phase {phase}")

        # Phase boundary
        if offset > 0:
            for ax in axes:
                ax.axvline(offset + 0.5, color="gray", linestyle=":", linewidth=1)

        offset += len(ph)

    axes[0].set_title("Loss (solid=train, dashed=val)")
    axes[0].set_xlabel("Epoch (global)")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Val Macro F1")
    axes[1].set_xlabel("Epoch (global)")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"training_curves_{model_label.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {fname}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(csv_path: str, save_dir: str):
    df = pd.read_csv(csv_path)
    df = df[df["aspect"] != "MACRO"]

    models  = df["model"].unique()
    aspects = [a for a in ASPECT_COLS if a in df["aspect"].values]
    shorts  = [ASPECT_SHORT.get(a, a) for a in aspects]

    x      = np.arange(len(aspects))
    width  = 0.35
    n      = len(models)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["f1", "precision", "recall"]
    titles  = ["F1 Score", "Precision", "Recall"]

    for ax, metric, title in zip(axes, metrics, titles):
        for i, model in enumerate(models):
            sub = df[df["model"] == model].set_index("aspect")
            vals = [sub.loc[a, metric] if a in sub.index else 0 for a in aspects]
            bars = ax.bar(x + offsets[i], vals, width,
                          label=model,
                          color=COLORS.get(model, f"C{i}"),
                          alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(shorts, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Add macro F1 text box
    macro_df = pd.read_csv(csv_path)
    macro_df = macro_df[macro_df["aspect"] == "MACRO"]
    box_lines = ["Macro F1"]
    for _, row in macro_df.iterrows():
        box_lines.append(f"  {row['model']}: {row['f1']:.3f}")
    box_text = "\n".join(box_lines)
    fig.text(0.5, -0.02, box_text, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Aspect Classification: Baseline vs Modern-FinBERT-large",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    fname = os.path.join(save_dir, "comparison_chart.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {fname}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    d = args.output_dir

    for name, label in [("baseline", "Baseline (FinBERT)"),
                         ("finetune", "Modern-FinBERT-large")]:
        log = os.path.join(d, f"{name}_training_log.json")
        if os.path.exists(log):
            plot_training_curves(log, label, d)

    csv = os.path.join(d, "comparison_results.csv")
    if os.path.exists(csv):
        plot_comparison(csv, d)
