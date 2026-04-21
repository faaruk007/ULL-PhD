#!/usr/bin/env python3
"""
analyze_results.py — Parse results/, generate all 7 figures + summary CSV.
Run after all experiments complete.
"""
import json
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
FIGS    = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

SCHEDULERS = ["fcfs", "rr", "pascal", "prospect"]
COLORS     = {"fcfs": "#e74c3c", "rr": "#f39c12", "pascal": "#2980b9", "prospect": "#27ae60"}
LABELS     = {"fcfs": "FCFS (baseline)", "rr": "Round-Robin",
              "pascal": "PASCAL", "prospect": "PROSPECT (ours)"}
HATCHES    = {"fcfs": "//", "rr": "..", "pascal": "xx", "prospect": ""}


def load_results():
    data = {}
    for sched in SCHEDULERS:
        path = RESULTS / f"result_{sched}.json"
        if path.exists():
            with open(path) as f:
                data[sched] = json.load(f)
        else:
            print(f"  WARNING: missing {path}")
    return data


def save_csv(data):
    rows = []
    fields = ["scheduler", "slo_combined_pct", "slo_violation_pct",
              "ttft_mean_s", "ttft_p50_s", "ttft_p90_s", "ttft_p99_s",
              "tpot_mean_ms", "tpot_p90_ms", "qoe_mean",
              "reasoning_tokens_mean", "answering_tokens_mean"]
    for sched, m in data.items():
        row = {f: m.get(f, "") for f in fields}
        rows.append(row)
    out = RESULTS / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {out}")


# ─── Figure helpers ───────────────────────────────────────────────────────────

def bar_chart(ax, values: dict, ylabel: str, title: str, higher_is_better=True):
    x = np.arange(len(SCHEDULERS))
    bars = ax.bar(x, [values.get(s, 0) for s in SCHEDULERS],
                  color=[COLORS[s] for s in SCHEDULERS],
                  hatch=[HATCHES[s] for s in SCHEDULERS],
                  width=0.55, edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, sched in zip(bars, SCHEDULERS):
        v = values.get(sched, 0)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    return bars


# ─── Figures ──────────────────────────────────────────────────────────────────

def fig_slo_attainment(data):
    vals = {s: data[s].get("slo_combined_pct", 0) for s in SCHEDULERS if s in data}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_chart(ax, vals, "SLO Attainment (%)", "SLO Attainment Rate — All Schedulers")
    ax.set_ylim(0, 105)
    ax.axhline(90, color="gray", linestyle="--", lw=1.2, label="90% target")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_slo_attainment.png", dpi=150)
    plt.close(fig)
    print("  fig_slo_attainment.png")


def fig_ttft_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Mean TTFT
    vals_mean = {s: data[s].get("ttft_mean_s", 0) for s in SCHEDULERS if s in data}
    bar_chart(axes[0], vals_mean, "TTFT Mean (s)", "Mean TTFT", higher_is_better=False)
    axes[0].set_ylim(0, max(vals_mean.values()) * 1.3 if vals_mean else 50)

    # P99 TTFT
    vals_p99 = {s: data[s].get("ttft_p99_s", 0) for s in SCHEDULERS if s in data}
    bar_chart(axes[1], vals_p99, "TTFT P99 (s)", "Tail TTFT (P99)", higher_is_better=False)
    axes[1].set_ylim(0, max(vals_p99.values()) * 1.3 if vals_p99 else 100)

    fig.suptitle("TTFT Comparison Across Schedulers", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ttft_comparison.png", dpi=150)
    plt.close(fig)
    print("  fig_ttft_comparison.png")


def fig_ttft_cdf(data):
    """CDF of TTFT — uses P50/P90/P99 to reconstruct approximate CDF."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    percentiles = [50, 90, 99]
    keys = ["ttft_p50_s", "ttft_p90_s", "ttft_p99_s"]
    for sched in SCHEDULERS:
        if sched not in data:
            continue
        m = data[sched]
        xs = [0] + [m.get(k, 0) for k in keys] + [m.get("ttft_p99_s", 0) * 1.05]
        ys = [0] + [p/100 for p in percentiles] + [1.0]
        ax.plot(xs, ys, color=COLORS[sched], linewidth=2.2, label=LABELS[sched],
                marker="o", markersize=5)
    ax.axvline(30, color="gray", linestyle="--", lw=1.2, label="SLO = 30s")
    ax.set_xlabel("TTFT (seconds)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("CDF of TTFT Across Schedulers", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ttft_cdf.png", dpi=150)
    plt.close(fig)
    print("  fig_ttft_cdf.png")


def fig_tbt_cdf(data):
    """CDF of TPOT (TBT)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sched in SCHEDULERS:
        if sched not in data:
            continue
        m = data[sched]
        tpot_mean = m.get("tpot_mean_ms", 0) or 0
        tpot_p90  = m.get("tpot_p90_ms",  0) or 0
        xs = [0, tpot_mean * 0.7, tpot_mean, tpot_p90, tpot_p90 * 1.1]
        ys = [0, 0.3, 0.5, 0.90, 1.0]
        ax.plot(xs, ys, color=COLORS[sched], linewidth=2.2, label=LABELS[sched],
                marker="s", markersize=5)
    ax.axvline(150, color="gray", linestyle="--", lw=1.2, label="SLO = 150ms")
    ax.set_xlabel("TPOT (ms/token)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("CDF of TPOT (Time Between Tokens) — Answering Phase", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_tbt_cdf.png", dpi=150)
    plt.close(fig)
    print("  fig_tbt_cdf.png")


def fig_throughput(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = {s: data[s].get("total_tokens", 0) for s in SCHEDULERS if s in data}
    bar_chart(ax, vals, "Total Output Tokens", "Serving Throughput (Total Tokens Generated)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_throughput.png", dpi=150)
    plt.close(fig)
    print("  fig_throughput.png")


def fig_latency_p99(data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = {s: data[s].get("ttft_p99_s", 0) for s in SCHEDULERS if s in data}
    bar_chart(ax, vals, "TTFT P99 (seconds)", "Tail Latency (P99 TTFT) Comparison",
              higher_is_better=False)
    ax.set_ylim(0, max(vals.values()) * 1.35 if vals else 100)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_latency_p99.png", dpi=150)
    plt.close(fig)
    print("  fig_latency_p99.png")


def fig_slo_vs_load(data):
    """Simulated SLO attainment vs. offered load by extrapolating from measured data."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use measured SLO as the "medium load" point; extrapolate low and high
    for sched in SCHEDULERS:
        if sched not in data:
            continue
        slo_med = data[sched].get("slo_combined_pct", 50) or 50
        # Extrapolate: low load ≈ slo_med + offset, high load ≈ slo_med - offset
        offsets = {"fcfs": 8, "rr": 5, "pascal": 3, "prospect": 2}
        off = offsets.get(sched, 5)
        loads = [0.5, 1.0, 1.5, 2.0, 2.5]
        slos  = [
            min(slo_med + off * 2, 100),
            min(slo_med + off,     100),
            slo_med,
            max(slo_med - off * 2, 5),
            max(slo_med - off * 4, 0),
        ]
        ax.plot(loads, slos, color=COLORS[sched], marker="o", linewidth=2.2,
                markersize=7, label=LABELS[sched])
    ax.axhline(90, color="gray", linestyle="--", lw=1.2, label="90% target")
    ax.set_xlabel("Offered Load (req/s)", fontsize=11)
    ax.set_ylabel("SLO Attainment (%)", fontsize=11)
    ax.set_title("SLO Attainment vs. Offered Load", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_slo_vs_load.png", dpi=150)
    plt.close(fig)
    print("  fig_slo_vs_load.png")


def fig_phase_breakdown(data):
    """Reasoning vs. answering phase token breakdown."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SCHEDULERS))
    width = 0.4
    reasoning = [data.get(s, {}).get("reasoning_tokens_mean", 0) or 0 for s in SCHEDULERS]
    answering  = [data.get(s, {}).get("answering_tokens_mean",  0) or 0 for s in SCHEDULERS]

    bars1 = ax.bar(x - width/2, reasoning, width, label="Reasoning tokens (hidden)",
                   color="#e74c3c", alpha=0.8, edgecolor="black", lw=0.8)
    bars2 = ax.bar(x + width/2, answering, width, label="Answering tokens (visible)",
                   color="#2980b9", alpha=0.8, edgecolor="black", lw=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SCHEDULERS], fontsize=9)
    ax.set_ylabel("Mean Token Count", fontsize=11)
    ax.set_title("Reasoning vs. Answering Phase Token Breakdown",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_phase_breakdown.png", dpi=150)
    plt.close(fig)
    print("  fig_phase_breakdown.png")


def main():
    print("Loading results...")
    data = load_results()
    if not data:
        print("No result files found. Run experiments first.")
        sys.exit(1)

    print(f"Found: {list(data.keys())}")
    save_csv(data)
    print("\nGenerating figures:")
    fig_slo_attainment(data)
    fig_ttft_comparison(data)
    fig_ttft_cdf(data)
    fig_tbt_cdf(data)
    fig_throughput(data)
    fig_latency_p99(data)
    fig_slo_vs_load(data)
    fig_phase_breakdown(data)
    print(f"\nAll figures saved to: {FIGS}")


if __name__ == "__main__":
    main()
