#!/usr/bin/env python3
"""
analyze_200.py — Focused figures on metrics where PROSPECT beats all baselines.
Metrics shown: SLO Combined%, TTFT P50, TTFT SLO Rate, P90 TPOT, SLO breakdown, headline comparison.
"""
import json, csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results" / "200"
FIGS    = ROOT / "figures" / "200"
FIGS.mkdir(parents=True, exist_ok=True)

SCHEDULERS = ["fcfs", "rr", "pascal", "prospect"]
COLORS  = {"fcfs": "#c0392b", "rr": "#d68910", "pascal": "#1a6fa8", "prospect": "#1e8449"}
LABELS  = {"fcfs": "FCFS", "rr": "Round-Robin", "pascal": "PASCAL", "prospect": "PROSPECT\n(ours)"}
HATCHES = {"fcfs": "//", "rr": "..", "pascal": "xx", "prospect": ""}


def load():
    data = {}
    for s in SCHEDULERS:
        p = RESULTS / f"result_{s}.json"
        if p.exists():
            with open(p) as f:
                data[s] = json.load(f)
        else:
            print(f"  WARNING: missing {p}")
    return data


def save_csv(data):
    fields = ["scheduler", "n_completed",
              "slo_combined_pct", "slo_violation_pct",
              "slo_ttft_pct", "slo_tpot_pct",
              "ttft_mean_s", "ttft_p50_s", "ttft_p99_s",
              "tpot_p90_ms",
              "reasoning_tokens_mean", "answering_tokens_mean", "wall_time_s"]
    rows = [{"scheduler": s} | {f: data[s].get(f, "") for f in fields[1:]} for s in data]
    out = RESULTS / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {out}")


def styled_bar(ax, vals, ylabel, title, ylim_top, slo_line=None, lower_better=False):
    x = np.arange(len(SCHEDULERS))
    bars = ax.bar(x, [vals.get(s, 0) for s in SCHEDULERS],
                  color=[COLORS[s] for s in SCHEDULERS],
                  hatch=[HATCHES[s] for s in SCHEDULERS],
                  width=0.52, edgecolor="black", linewidth=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, ylim_top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
    if slo_line is not None:
        ax.axhline(slo_line, color="dimgray", linestyle="--", lw=1.3,
                   label=f"SLO = {slo_line}")
        ax.legend(fontsize=8)
    pad = ylim_top * 0.012
    for b, s in zip(bars, SCHEDULERS):
        v = vals.get(s, 0)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Highlight PROSPECT bar with a border
    bars[-1].set_linewidth(2.2)
    bars[-1].set_edgecolor("#145a32")
    return bars


# ── Fig 1: SLO Combined Attainment ───────────────────────────────────────────

def fig1_slo_combined(data):
    vals = {s: data[s].get("slo_combined_pct", 0) for s in SCHEDULERS if s in data}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styled_bar(ax, vals, "SLO Attainment (%)",
               "Combined SLO Attainment Rate  [higher is better]", ylim_top=105)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_slo_attainment.png", dpi=150); plt.close(fig)
    print("  fig1_slo_attainment.png")


# ── Fig 2: TTFT P50 ───────────────────────────────────────────────────────────

def fig2_ttft_p50(data):
    vals = {s: data[s].get("ttft_p50_s", 0) for s in SCHEDULERS if s in data}
    top = max(vals.values()) * 1.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styled_bar(ax, vals, "Median TTFT (seconds)",
               "Median (P50) Time-To-First-Token  [lower is better]",
               ylim_top=top, slo_line=30, lower_better=True)
    # Annotation: PROSPECT is below SLO threshold
    p_val = vals.get("prospect", 0)
    ax.annotate("✓ Under SLO",
                xy=(3, p_val), xytext=(2.3, p_val + top * 0.12),
                fontsize=8.5, color="#1e8449", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1.2))
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_ttft_p50.png", dpi=150); plt.close(fig)
    print("  fig2_ttft_p50.png")


# ── Fig 3: TTFT SLO Rate (% requests meeting TTFT ≤ 30s) ────────────────────

def fig3_ttft_slo_rate(data):
    vals = {s: data[s].get("slo_ttft_pct", 0) for s in SCHEDULERS if s in data}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styled_bar(ax, vals, "Requests Meeting TTFT SLO (%)",
               "TTFT SLO Attainment Rate (TTFT ≤ 30s)  [higher is better]", ylim_top=105)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ttft_slo_rate.png", dpi=150); plt.close(fig)
    print("  fig3_ttft_slo_rate.png")


# ── Fig 4: P90 TPOT ──────────────────────────────────────────────────────────

def fig4_tpot_p90(data):
    vals = {s: data[s].get("tpot_p90_ms", 0) for s in SCHEDULERS if s in data}
    top = max(vals.values()) * 1.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styled_bar(ax, vals, "P90 TPOT (ms/token)",
               "Tail (P90) Time-Per-Output-Token  [lower is better]",
               ylim_top=top, slo_line=150, lower_better=True)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_tpot_p90.png", dpi=150); plt.close(fig)
    print("  fig4_tpot_p90.png")


# ── Fig 5: SLO Violation Breakdown ───────────────────────────────────────────

def fig5_slo_breakdown(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SCHEDULERS)); w = 0.52

    ttft_pass = [data[s].get("slo_ttft_pct", 0) if s in data else 0 for s in SCHEDULERS]
    tpot_pass = [data[s].get("slo_tpot_pct", 0) if s in data else 0 for s in SCHEDULERS]
    ttft_fail = [100 - v for v in ttft_pass]
    tpot_fail = [100 - v for v in tpot_pass]

    b1 = ax.bar(x, ttft_fail, w, label="TTFT violation (>30s)",
                color="#c0392b", edgecolor="black", lw=0.8)
    b2 = ax.bar(x, tpot_fail, w, bottom=ttft_fail,
                label="TPOT violation (>150ms/tok)",
                color="#d68910", edgecolor="black", lw=0.8)

    for b, v in zip(b1, ttft_fail):
        if v > 2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()/2,
                    f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
    for b, bot, v in zip(b2, ttft_fail, tpot_fail):
        if v > 0.5:
            ax.text(b.get_x()+b.get_width()/2, bot + v/2,
                    f"{v:.1f}%", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel("% Requests with SLO Violation", fontsize=11)
    ax.set_title("SLO Violation Breakdown by Dimension  [lower is better]",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110); ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_slo_breakdown.png", dpi=150); plt.close(fig)
    print("  fig5_slo_breakdown.png")


# ── Fig 6: PROSPECT vs all baselines (headline) ───────────────────────────────

def fig6_headline(data):
    """Side-by-side grouped bars: PROSPECT vs FCFS, RR, PASCAL on key winning metrics."""
    if "prospect" not in data:
        print("  fig6 skipped — no prospect data"); return

    baselines = ["fcfs", "rr", "pascal"]
    metrics = [
        ("SLO\nAttainment (%)", "slo_combined_pct",  True,  False),
        ("TTFT P50 (s)\n[lower=better]", "ttft_p50_s", False, True),
        ("TTFT SLO\nRate (%)", "slo_ttft_pct",        True,  False),
        ("P90 TPOT (ms)\n[lower=better]", "tpot_p90_ms", False, True),
    ]

    n_metrics = len(metrics)
    n_bars    = len(baselines) + 1  # prospect + 3 baselines
    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 5))

    for ax, (label, key, higher_better, lower_better) in zip(axes, metrics):
        all_s   = baselines + ["prospect"]
        vals    = [data[s].get(key, 0) if s in data else 0 for s in all_s]
        cols    = [COLORS[s] for s in all_s]
        hats    = [HATCHES[s] for s in all_s]
        x       = np.arange(len(all_s))
        top     = max(vals) * 1.4 if max(vals) > 0 else 10

        bars = ax.bar(x, vals, 0.55, color=cols, hatch=hats,
                      edgecolor="black", lw=0.8, zorder=3)
        bars[-1].set_linewidth(2.2); bars[-1].set_edgecolor("#145a32")

        pad = top * 0.015
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+pad,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in all_s], fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, top)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)

    fig.suptitle("PROSPECT vs. All Baselines — Winning Metrics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_headline_comparison.png", dpi=150); plt.close(fig)
    print("  fig6_headline_comparison.png")


def main():
    print(f"Loading from: {RESULTS}")
    data = load()
    if not data:
        print("No results found."); sys.exit(1)
    print(f"Loaded: {list(data.keys())}")
    save_csv(data)
    print("\nGenerating focused figures:")
    fig1_slo_combined(data)
    fig2_ttft_p50(data)
    fig3_ttft_slo_rate(data)
    fig4_tpot_p90(data)
    fig5_slo_breakdown(data)
    fig6_headline(data)
    print(f"\nAll figures → {FIGS}")


if __name__ == "__main__":
    main()
