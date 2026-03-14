"""
plot_ablation.py
----------------
Visualise ablation_results.csv produced by test_ablation_rmse.py.

Usage:
    python plot_ablation.py [--csv ablation_results.csv] [--out ablation_plots.png]

Panels:
  1. act_rmse          — end-task sensitivity per config
  2. act_snr_db        — signal-to-noise ratio (hardware unit)
  3. max_act_err       — worst-case joint error (safety)
  4. act_cos_sim       — directional fidelity of actions
  5. avg_rel_rmse      — per-layer quantization quality
  6. pct_layers_above_5 — % layers with >5% error
  7. per-component rel_rmse — which layers drive error (grouped bars)
  8. per-joint RMSE   — which DOFs are most affected (3C1 configs only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Aesthetics ────────────────────────────────────────────────────────────────

PALETTE = {
    "vision":       "#4e79a7",
    "transformer":  "#f28e2b",
    "action_head":  "#59a14f",
    "multi":        "#b07aa1",
}

COMP_COLORS = {
    "rel_vision":       "#4e79a7",
    "rel_language":     "#f28e2b",
    "rel_action_expert":"#e15759",
    "rel_action_head":  "#59a14f",
}

JOINT_LABELS = [f"J{i}" for i in range(7)] + ["grip"]


def _bar_color(label: str) -> str:
    comps = label.split("+")
    if len(comps) == 1:
        return PALETTE.get(comps[0], "#aec7e8")
    return PALETTE["multi"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hbars(ax, df: pd.DataFrame, col: str, title: str, xlabel: str,
           log: bool = False) -> None:
    """Horizontal bar chart for one scalar metric across all configs."""
    colors = [_bar_color(lbl) for lbl in df["config"]]
    vals = df[col].values
    y = np.arange(len(df))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["config"], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def _grouped_comp_bars(ax, df: pd.DataFrame) -> None:
    """Grouped bars: per-component rel_rmse for each config."""
    comp_cols = ["rel_vision", "rel_language", "rel_action_expert", "rel_action_head"]
    comp_labels = ["vision", "language", "act_expert", "act_head"]
    n_cfg = len(df)
    n_comp = len(comp_cols)
    width = 0.18
    x = np.arange(n_cfg)
    for i, (col, lbl) in enumerate(zip(comp_cols, comp_labels)):
        vals = pd.to_numeric(df[col], errors="coerce").values
        offset = (i - n_comp / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=lbl,
               color=COMP_COLORS[col], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("rel_rmse", fontsize=8)
    ax.set_title("Per-component relative RMSE", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)


def _joint_rmse_bars(ax, df: pd.DataFrame) -> None:
    """Per-joint RMSE for single-component configs (3C1 only)."""
    single = df[~df["config"].str.contains(r"\+")]
    joint_cols = [f"joint_{i}_rmse" for i in range(8)]
    n_cfg = len(single)
    n_joint = len(joint_cols)
    width = 0.25
    x = np.arange(n_joint)
    for i, (_, row) in enumerate(single.iterrows()):
        vals = [pd.to_numeric(row[c], errors="coerce") for c in joint_cols]
        offset = (i - n_cfg / 2 + 0.5) * width
        color = _bar_color(row["config"])
        ax.bar(x + offset, vals, width, label=row["config"],
               color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_LABELS, fontsize=8)
    ax.set_ylabel("act_rmse per joint", fontsize=8)
    ax.set_title("Per-joint action RMSE (single-component configs)", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)


# ── Legend patch ─────────────────────────────────────────────────────────────

def _add_legend(fig) -> None:
    from matplotlib.patches import Patch
    handles = [
        Patch(color=PALETTE["vision"],      label="vision"),
        Patch(color=PALETTE["transformer"], label="transformer"),
        Patch(color=PALETTE["action_head"], label="action_head"),
        Patch(color=PALETTE["multi"],       label="multi-component"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="ablation_results.csv")
    p.add_argument("--out", default="ablation_plots.png")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run test_ablation_rmse.py first."
        )

    df = pd.read_csv(csv_path)
    # rename column if needed (CSV uses action_expert, plot expects action_expert)
    if "rel_action_expert" not in df.columns and "rel_action_expert" not in df.columns:
        df = df.rename(columns={"rel_act_exp": "rel_action_expert"})

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"Pi0 Quantization Ablation  —  {len(df)} configs  "
        f"(FP8 E4M3 in / FP16 out)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # 2-row, 4-col grid; bottom row uses 2 wider panels
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.93, bottom=0.10)

    # Row 0: 4 scalar metrics
    ax_rmse   = fig.add_subplot(gs[0, 0])
    ax_snr    = fig.add_subplot(gs[0, 1])
    ax_maxerr = fig.add_subplot(gs[0, 2])
    ax_cos    = fig.add_subplot(gs[0, 3])

    _hbars(ax_rmse,   df, "act_rmse",    "Action RMSE",           "RMSE (joint units)")
    _hbars(ax_snr,    df, "act_snr_db",  "Action SNR",            "dB")
    _hbars(ax_maxerr, df, "max_act_err", "Max action error",      "joint units", log=True)
    _hbars(ax_cos,    df, "act_cos_sim", "Action cosine sim",     "cosine similarity")

    # Row 1: 2 wider panels
    ax_rel  = fig.add_subplot(gs[1, 0:2])
    ax_pct  = fig.add_subplot(gs[1, 2])
    ax_jnt  = fig.add_subplot(gs[1, 3])

    _grouped_comp_bars(ax_rel, df)
    _hbars(ax_pct, df, "pct_layers_above_5",
           "Layers with >5% error", "% of quantized layers")
    _joint_rmse_bars(ax_jnt, df)

    _add_legend(fig)

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
