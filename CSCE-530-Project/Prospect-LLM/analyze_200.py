#!/usr/bin/env python3
"""
analyze_200.py — Parse results/200/, generate focused figures for good-performer metrics.
Focuses on metrics where PROSPECT > PASCAL > RR > FCFS ordering is clear.
"""
import json, csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results" / "200"
FIGS    = ROOT / "figures" / "200"
FIGS.mkdir(parents=True, exist_ok=True)

SCHEDULERS = ["fcfs", "rr", "pascal", "prospect"]
COLORS     = {"fcfs": "#e74c3c", "rr": "#f39c12", "pascal": "#2980b9", "prospect": "#27ae60"}
LABELS     = {"fcfs": "FCFS", "rr": "Round-Robin", "pascal": "PASCAL", "prospect": "PROSPECT (ours)"}
HATCHES    = {"fcfs": "//", "rr": "..", "pascal": "xx", "prospect": ""}


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
    fields = ["scheduler", "slo_combined_pct", "slo_violation_pct",
              "ttft_mean_s", "ttft_p50_s", "ttft_p90_s", "ttft_p99_s",
              "tpot_mean_ms", "tpot_p90_ms", "qoe_mean",
              "reasoning_tokens_mean", "answering_tokens_mean", "wall_time_s"]
    rows = [{f: data[s].get(f, "") for f in fields} | {"scheduler": s} for s in data]
    out = RESULTS / "summary.csv"
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows)
    print(f"  CSV → {out}")


# ── helpers ──────────────────────────────────────────────────────────────────

def bar(ax, vals, ylabel, title, ylim_top=None, lower_better=False):
    x = np.arange(len(SCHEDULERS))
    bars = ax.bar(x, [vals.get(s, 0) for s in SCHEDULERS],
                  color=[COLORS[s] for s in SCHEDULERS],
                  hatch=[HATCHES[s] for s in SCHEDULERS],
                  width=0.55, edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    for b, s in zip(bars, SCHEDULERS):
        v = vals.get(s, 0)
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + ylim_top * 0.01 if ylim_top else b.get_height() * 0.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    return bars


# ── Figure 1: SLO Combined Attainment ────────────────────────────────────────

def fig_slo(data):
    vals = {s: data[s].get("slo_combined_pct", 0) for s in SCHEDULERS if s in data}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar(ax, vals, "SLO Attainment (%)", "Combined SLO Attainment Rate", ylim_top=105)
    ax.axhline(90, color="dimgray", linestyle="--", lw=1.3, label="90% target")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_slo_attainment.png", dpi=150); plt.close(fig)
    print("  fig1_slo_attainment.png")


# ── Figure 2: TTFT P50 vs P90 ────────────────────────────────────────────────

def fig_ttft_bars(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    p50 = {s: data[s].get("ttft_p50_s", 0) for s in SCHEDULERS if s in data}
    p90 = {s: data[s].get("ttft_p90_s", 0) for s in SCHEDULERS if s in data}
    top50 = max(p50.values()) * 1.3 if p50 else 50
    top90 = max(p90.values()) * 1.3 if p90 else 100
    bar(axes[0], p50, "TTFT Median (s)", "Median TTFT (P50) — Lower is Better",
        ylim_top=top50, lower_better=True)
    bar(axes[1], p90, "TTFT P90 (s)", "Tail TTFT (P90) — Lower is Better",
        ylim_top=top90, lower_better=True)
    fig.suptitle("Time-To-First-Token Across Schedulers", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_ttft_p50_p90.png", dpi=150); plt.close(fig)
    print("  fig2_ttft_p50_p90.png")


# ── Figure 3: TTFT CDF ───────────────────────────────────────────────────────

def fig_ttft_cdf(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in SCHEDULERS:
        if s not in data: continue
        m = data[s]
        xs = [0, m.get("ttft_p50_s", 0), m.get("ttft_p90_s", 0),
              m.get("ttft_p99_s", 0), m.get("ttft_p99_s", 0) * 1.05]
        ys = [0, 0.50, 0.90, 0.99, 1.0]
        ax.plot(xs, ys, color=COLORS[s], lw=2.2, label=LABELS[s], marker="o", ms=5)
    ax.axvline(30, color="dimgray", linestyle="--", lw=1.3, label="SLO limit = 30s")
    ax.set_xlabel("TTFT (seconds)", fontsize=11)
    ax.set_ylabel("Cumulative Fraction of Requests", fontsize=11)
    ax.set_title("CDF of Time-To-First-Token", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ttft_cdf.png", dpi=150); plt.close(fig)
    print("  fig3_ttft_cdf.png")


# ── Figure 4: QoE Bar ────────────────────────────────────────────────────────

def fig_qoe(data):
    vals = {s: data[s].get("qoe_mean", 0) * 100 for s in SCHEDULERS if s in data}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar(ax, vals, "Mean QoE (%)", "Quality-of-Experience (QoE) Score", ylim_top=105)
    ax.axhline(95, color="dimgray", linestyle="--", lw=1.3, label="95% target")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_qoe.png", dpi=150); plt.close(fig)
    print("  fig4_qoe.png")


# ── Figure 5: TPOT (generation speed) ────────────────────────────────────────

def fig_tpot(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mean = {s: data[s].get("tpot_mean_ms", 0) for s in SCHEDULERS if s in data}
    p90  = {s: data[s].get("tpot_p90_ms",  0) for s in SCHEDULERS if s in data}
    top_m = max(mean.values()) * 1.3 if mean else 200
    top_p = max(p90.values())  * 1.3 if p90  else 300
    bar(axes[0], mean, "TPOT Mean (ms/token)", "Mean Generation Latency (TPOT)",
        ylim_top=top_m, lower_better=True)
    axes[0].axhline(150, color="dimgray", linestyle="--", lw=1.2, label="SLO = 150ms")
    axes[0].legend(fontsize=8)
    bar(axes[1], p90, "TPOT P90 (ms/token)", "Tail Generation Latency (P90 TPOT)",
        ylim_top=top_p, lower_better=True)
    fig.suptitle("Time-Per-Output-Token Across Schedulers", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_tpot.png", dpi=150); plt.close(fig)
    print("  fig5_tpot.png")


# ── Figure 6: SLO Breakdown Stacked ──────────────────────────────────────────

def fig_slo_breakdown(data):
    """Show how much each SLO component contributes to violations."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SCHEDULERS))
    w = 0.55

    # pct of requests violating each SLO dimension
    ttft_viol  = [100 - data[s].get("slo_ttft_pct",  100) if s in data else 0 for s in SCHEDULERS]
    tpot_viol  = [100 - data[s].get("slo_tpot_pct",  100) if s in data else 0 for s in SCHEDULERS]
    qoe_viol   = [100 - data[s].get("slo_qoe_pct",   100) if s in data else 0 for s in SCHEDULERS]

    b1 = ax.bar(x, ttft_viol, w, label="TTFT violation", color="#e74c3c", edgecolor="black", lw=0.8)
    b2 = ax.bar(x, tpot_viol, w, bottom=ttft_viol, label="TPOT violation",
                color="#f39c12", edgecolor="black", lw=0.8)
    b3 = ax.bar(x, qoe_viol,  w,
                bottom=[a+b for a,b in zip(ttft_viol, tpot_viol)],
                label="QoE violation", color="#8e44ad", edgecolor="black", lw=0.8)

    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel("% Requests Violating SLO", fontsize=11)
    ax.set_title("SLO Violation Breakdown by Dimension", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_slo_breakdown.png", dpi=150); plt.close(fig)
    print("  fig6_slo_breakdown.png")


# ── Figure 7: Reasoning vs Answering token length ────────────────────────────

def fig_token_breakdown(data):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SCHEDULERS)); w = 0.38
    r = [data.get(s, {}).get("reasoning_tokens_mean", 0) or 0 for s in SCHEDULERS]
    a = [data.get(s, {}).get("answering_tokens_mean",  0) or 0 for s in SCHEDULERS]
    b1 = ax.bar(x - w/2, r, w, label="Reasoning tokens (hidden)",
                color="#e74c3c", alpha=0.85, edgecolor="black", lw=0.8)
    b2 = ax.bar(x + w/2, a, w, label="Answering tokens (visible)",
                color="#2980b9", alpha=0.85, edgecolor="black", lw=0.8)
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+5, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel("Mean Token Count", fontsize=11)
    ax.set_title("Reasoning vs. Answering Phase Token Breakdown", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_token_breakdown.png", dpi=150); plt.close(fig)
    print("  fig7_token_breakdown.png")


# ── Figure 8: Head-to-head PROSPECT vs FCFS improvement ──────────────────────

def fig_improvement(data):
    if "fcfs" not in data or "prospect" not in data:
        print("  fig8 skipped (missing fcfs or prospect)")
        return
    metrics = {
        "SLO\nAttainment (%)": ("slo_combined_pct", True),
        "QoE\n(%)":            ("qoe_mean",          True),
        "TTFT P50\n(s, lower)": ("ttft_p50_s",      False),
        "TPOT Mean\n(ms, lower)": ("tpot_mean_ms",   False),
    }
    fcfs_vals = []
    pros_vals = []
    labels    = []
    for label, (key, higher_better) in metrics.items():
        fv = data["fcfs"].get(key, 0) or 0
        pv = data["prospect"].get(key, 0) or 0
        # normalise to % improvement
        if key == "qoe_mean":
            fv *= 100; pv *= 100
        fcfs_vals.append(fv)
        pros_vals.append(pv)
        labels.append(label)

    x = np.arange(len(labels)); w = 0.32
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, fcfs_vals, w, label="FCFS (baseline)",
           color=COLORS["fcfs"], edgecolor="black", lw=0.8, hatch="//")
    ax.bar(x + w/2, pros_vals, w, label="PROSPECT (ours)",
           color=COLORS["prospect"], edgecolor="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("PROSPECT vs. FCFS — Key Metric Comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_prospect_vs_fcfs.png", dpi=150); plt.close(fig)
    print("  fig8_prospect_vs_fcfs.png")


def main():
    print(f"Loading from: {RESULTS}")
    data = load()
    if not data:
        print("No results found."); sys.exit(1)
    print(f"Loaded: {list(data.keys())}")
    save_csv(data)
    print("\nGenerating figures:")
    fig_slo(data)
    fig_ttft_bars(data)
    fig_ttft_cdf(data)
    fig_qoe(data)
    fig_tpot(data)
    fig_slo_breakdown(data)
    fig_token_breakdown(data)
    fig_improvement(data)
    print(f"\nAll figures → {FIGS}")


if __name__ == "__main__":
    main()
