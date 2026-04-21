#!/usr/bin/env python3
"""
run_all_experiments.py — Run all 4 schedulers sequentially on single GPU.

Usage:
    python3 run_all_experiments.py [--scheduler fcfs|rr|pascal|prospect|all]
                                   [--num-requests N]
                                   [--results-dir PATH]
"""
import argparse
import os
import sys
from pathlib import Path

# Set env before importing vLLM
os.environ["HF_HOME"]          = "/ddnB/work/faaruk/.cache/huggingface"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HF_TOKEN_HERE"
os.environ["TRANSFORMERS_CACHE"] = "/ddnB/work/faaruk/.cache/huggingface"
os.environ["XDG_CACHE_HOME"]    = "/ddnB/work/faaruk/.cache"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.scheduler_fcfs     import FCFSScheduler
from src.scheduler_rr       import RoundRobinScheduler
from src.scheduler_pascal   import PascalScheduler
from src.scheduler_prospect import ProspectScheduler
from src.serving_engine     import run_experiment

# --------------------------------------------------------------------------
MODEL     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
RESULTS   = str(ROOT / "results")
WORKLOAD  = str(ROOT / "results" / "workload_200.json")

SCHEDULER_CONFIG = {
    "token_quantum":           500,
    "demotion_threshold_tokens": 5000,
    "max_concurrent_long":     4,
}

ENGINE_CONFIG = dict(
    model_name            = MODEL,
    workload_path         = WORKLOAD,
    num_requests          = 200,
    arrival_rate          = 1.0,
    max_new_tokens        = 1024,
    max_concurrent        = 8,
    results_dir           = RESULTS,
    seed                  = 42,
    gpu_memory_utilization= 0.88,
    max_model_len         = 8192,
    dtype                 = "float16",
)

SCHEDULERS = {
    "fcfs":    FCFSScheduler,
    "rr":      RoundRobinScheduler,
    "pascal":  PascalScheduler,
    "prospect":ProspectScheduler,
}
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduler", default="all",
                    choices=list(SCHEDULERS) + ["all"])
    ap.add_argument("--num-requests", type=int, default=None,
                    help="Override number of requests (default: from ENGINE_CONFIG)")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Override results directory")
    args = ap.parse_args()

    cfg = dict(ENGINE_CONFIG)
    if args.num_requests is not None:
        cfg["num_requests"] = args.num_requests
        cfg["workload_path"] = str(ROOT / "results" / f"workload_{args.num_requests}.json")
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    to_run = list(SCHEDULERS.keys()) if args.scheduler == "all" else [args.scheduler]

    all_metrics = {}
    for name in to_run:
        cls = SCHEDULERS[name]
        print(f"\n{'#'*60}", flush=True)
        print(f"  Running: {name.upper()}", flush=True)
        print(f"{'#'*60}", flush=True)
        m = run_experiment(
            scheduler_cls=cls,
            scheduler_config=SCHEDULER_CONFIG,
            **cfg,
        )
        all_metrics[name] = m

    print("\n\n===== SUMMARY =====")
    header = f"{'Scheduler':<12} {'SLO%':>7} {'TTFT-mean':>10} {'TTFT-P90':>10} {'TPOT-mean':>10} {'QoE':>7}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(f"{name:<12} "
              f"{m.get('slo_combined_pct',0):>7.1f} "
              f"{m.get('ttft_mean_s','?'):>10} "
              f"{m.get('ttft_p90_s','?'):>10} "
              f"{m.get('tpot_mean_ms','?'):>10} "
              f"{m.get('qoe_mean','?'):>7}")


if __name__ == "__main__":
    main()
