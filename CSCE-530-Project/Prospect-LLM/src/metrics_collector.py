"""
metrics_collector.py — Compute and report all metrics from completed requests.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from .scheduler_base import RequestState


SLO_TTFT_S   = 30.0   # seconds
SLO_TPOT_MS  = 150.0  # ms/token
QOE_THRESHOLD = 0.95


def compute_metrics(requests: List[RequestState], scheduler_name: str) -> Dict[str, Any]:
    completed = [r for r in requests if r.completed]
    n = len(completed)
    if n == 0:
        return {"scheduler": scheduler_name, "n_completed": 0}

    # TTFT (time from arrival to first ANSWERING token = think_end_time)
    ttfts = [r.ttft for r in completed if r.ttft is not None]
    # TPOT
    tpots = [r.tpot_ms for r in completed if r.tpot_ms is not None]
    # QoE
    qoes  = [r.qoe for r in completed if r.qoe is not None]

    # Reasoning tokens
    reasoning_lens = [r.reasoning_tokens for r in completed]
    answering_lens  = [r.answering_tokens  for r in completed]

    # SLO attainment
    ttft_ok  = [t <= SLO_TTFT_S  for t in ttfts] if ttfts else []
    tpot_ok  = [t <= SLO_TPOT_MS for t in tpots] if tpots else []
    qoe_ok   = [q >= QOE_THRESHOLD for q in qoes] if qoes else []

    # Combined SLO: TTFT + TPOT + QoE all satisfied
    combined_ok = []
    for r in completed:
        ok = True
        if r.ttft is not None  and r.ttft  > SLO_TTFT_S:   ok = False
        if r.tpot_ms is not None and r.tpot_ms > SLO_TPOT_MS: ok = False
        if r.qoe is not None   and r.qoe   < QOE_THRESHOLD: ok = False
        combined_ok.append(ok)

    def pct(lst): return round(sum(lst) / len(lst) * 100, 2) if lst else None
    def safe_p(lst, p): return round(float(np.percentile(lst, p)), 3) if lst else None
    def safe_mean(lst): return round(float(np.mean(lst)), 3) if lst else None

    return {
        "scheduler": scheduler_name,
        "n_total": len(requests),
        "n_completed": n,

        # Primary metrics
        "ttft_mean_s":    safe_mean(ttfts),
        "ttft_p50_s":     safe_p(ttfts, 50),
        "ttft_p90_s":     safe_p(ttfts, 90),
        "ttft_p99_s":     safe_p(ttfts, 99),
        "tpot_mean_ms":   safe_mean(tpots),
        "tpot_p90_ms":    safe_p(tpots, 90),
        "qoe_mean":       safe_mean(qoes),

        # SLO attainment
        "slo_ttft_pct":     pct(ttft_ok),
        "slo_tpot_pct":     pct(tpot_ok),
        "slo_qoe_pct":      pct(qoe_ok),
        "slo_combined_pct": pct(combined_ok),
        "slo_violation_pct": round(100 - (pct(combined_ok) or 100), 2),

        # Phase breakdown
        "reasoning_tokens_mean": safe_mean(reasoning_lens),
        "reasoning_tokens_p90":  safe_p(reasoning_lens, 90),
        "answering_tokens_mean":  safe_mean(answering_lens),

        # Throughput
        "total_tokens": sum(r.total_output_tokens for r in completed),
    }


def save_results(metrics: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results → {path}")


def print_summary(metrics: Dict[str, Any]):
    s = metrics.get("scheduler", "?")
    print(f"\n{'='*55}")
    print(f"  Scheduler: {s.upper()}")
    print(f"  Completed: {metrics.get('n_completed')}/{metrics.get('n_total')}")
    print(f"  TTFT mean/P90/P99: "
          f"{metrics.get('ttft_mean_s')}s / "
          f"{metrics.get('ttft_p90_s')}s / "
          f"{metrics.get('ttft_p99_s')}s")
    print(f"  TPOT mean/P90:     "
          f"{metrics.get('tpot_mean_ms')}ms / "
          f"{metrics.get('tpot_p90_ms')}ms")
    print(f"  QoE mean:          {metrics.get('qoe_mean')}")
    print(f"  SLO Combined:      {metrics.get('slo_combined_pct')}%  "
          f"(violation: {metrics.get('slo_violation_pct')}%)")
    print(f"  Reasoning tok mean: {metrics.get('reasoning_tokens_mean')}")
    print(f"{'='*55}")
