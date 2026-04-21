# PASCAL & PROSPECT: Phase-Aware Scheduling for Reasoning LLMs

Implementation and evaluation of PASCAL (HPCA 2026) and PROSPECT (novel extension) on DeepSeek-R1-Distill-Qwen-7B using vLLM 0.5.0.

**Course:** CSCE 530 — Computer Architecture | **Platform:** LONI HPC, NVIDIA A100 80GB

---

## Results Summary (1000 Requests)

| Scheduler | SLO Combined | TTFT Mean | TTFT P50 | TTFT P90 | TTFT P99 | TPOT Mean | QoE |
|-----------|:-----------:|:---------:|:--------:|:--------:|:--------:|:---------:|:---:|
| FCFS (baseline) | 4.6% | 312.7s | 318.5s | 576.1s | 622.5s | 31.9ms | 0.984 |
| Round-Robin | 5.1% | 291.7s | 297.3s | 538.0s | 580.4s | 31.1ms | 0.984 |
| PASCAL | 5.1% | 289.7s | 295.2s | 534.4s | 576.9s | 31.0ms | 0.984 |
| **PROSPECT (ours)** | **10.6%** | 293.6s | **103.6s** | 934.4s | 1355.9s | **30.5ms** | 0.984 |

**PROSPECT: +6pp SLO over FCFS, 3× better median TTFT (P50: 103.6s vs 318.5s).**

> **Why is SLO% low overall?** With 1000 requests at 1 req/s over a 1031s window and 125 batches at ~13s each, most requests wait far beyond the 30s TTFT SLO regardless of scheduler. PROSPECT doubles the fraction meeting SLO by front-loading SHORT requests.

---

## Why PROSPECT Works

Reasoning LLMs (DeepSeek-R1) generate a hidden `<think>…</think>` chain before the visible answer. This means:

**TTFT = prefill time + full reasoning chain + first answer token**

SHORT requests (factual questions) have short chains (<512 tokens). LONG requests (math proofs) can have 2000+ tokens. FCFS/RR/PASCAL mix them randomly — SHORT requests wait behind LONG ones.

**PROSPECT** fixes this by predicting reasoning length at admission:

```
Request arrives
    │
    ▼
ORLP Predictor ─── keyword heuristics + prompt length ──► SHORT / MEDIUM / LONG
    │
    ▼
Admission Queue  ← sorted SHORT-first; LONG capped at 4 concurrent
    │
    ▼
vLLM Batch (8 concurrent) ──► Phase Detector (detects </think>)
    │
    ▼
Online Calibrator (updates bucket means from realized lengths)
```

SHORT requests (30% of workload) get served in early batches → TTFT ≈ 13–100s instead of waiting in the full 318s average queue. **Median TTFT drops 3×.**

---

## How It Works — Schedulers

| Scheduler | Mechanism |
|-----------|-----------|
| **FCFS** | Admit in arrival order. No awareness of request type or phase. |
| **Round-Robin** | Rotate every 500 tokens. Prevents monopolization, mild TTFT improvement. |
| **PASCAL** | Separate HIGH (reasoning) and LOW (answering) queues. Demote after 5000 reasoning tokens. |
| **PROSPECT** | Predict reasoning length → SHORT-first admission → cap LONG concurrency → online calibrate. |

PASCAL ≈ FCFS in this simulation because offline batch mode limits true phase-based preemption. PROSPECT's length-aware sorting is the decisive differentiator.

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | DeepSeek-R1-Distill-Qwen-7B |
| **Model size** | 7B parameters, 14.27 GB (float16) |
| **GPU** | NVIDIA A100 80GB (single GPU) |
| **Cluster** | LONI HPC, `gpu2` partition |
| **vLLM** | 0.5.0.post1, `enforce_eager=True` |
| **max_model_len** | 8192 tokens |
| **max_new_tokens** | 1024 tokens |
| **Batch size** | 8 concurrent requests |
| **gpu_memory_utilization** | 0.88 |
| **Requests** | 1000 (seed=42) |
| **Arrival pattern** | Poisson, λ=1.0 req/s, span≈1031s |
| **Workload mix** | 30% SHORT / 50% MEDIUM / 20% LONG |
| **Token quantum** | 500 (RR/PASCAL) |
| **Demotion threshold** | 5000 reasoning tokens (PASCAL) |
| **max_concurrent_long** | 4 (PROSPECT) |

### Workload / Dataset

Synthetic workload, 1000 prompts (seed=42, reproducible):

| Type | Count | Example Prompts |
|------|-------|-----------------|
| **SHORT** | 300 | "What is the capital of France?", "What is H₂O's formula?" |
| **MEDIUM** | 500 | "Explain gradient descent", "Write Sieve of Eratosthenes in Python" |
| **LONG** | 200 | "Prove infinitely many primes exist", "Derive FFT O(n log n) complexity" |

### SLO Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| TTFT | ≤ 30 s | Time to first visible answer token |
| TPOT | ≤ 150 ms/tok | Smooth token streaming (>6 tok/s) |
| QoE | ≥ 0.95 | Normalized delivery quality score |

---

## Setup & Run

```bash
# Requires: vLLM 0.5.0, Python 3.10, NVIDIA A100
conda activate vllm05_env

# Single scheduler, 1000 requests
python3 run_all_experiments.py --scheduler prospect --num-requests 1000

# All 4 schedulers sequentially
python3 run_all_experiments.py --scheduler all --num-requests 1000

# Generate 8 figures + CSV  (needs matplotlib — use jitserve_env)
/path/to/jitserve_env/bin/python3 analyze_results.py

# SLURM — 4 GPUs in parallel (~27 min total)
sbatch job1k_fcfs.slurm && sbatch job1k_rr.slurm \
  && sbatch job1k_pascal.slurm && sbatch job1k_prospect.slurm
```

---

## Project Structure

```
Prospect-LLM/
├── src/
│   ├── scheduler_base.py       # Base class, RequestState, Phase enum
│   ├── scheduler_fcfs.py       # FCFS baseline
│   ├── scheduler_rr.py         # Round-Robin (token quantum=500)
│   ├── scheduler_pascal.py     # PASCAL phase-aware scheduler
│   ├── scheduler_prospect.py   # PROSPECT prediction-based scheduler
│   ├── serving_engine.py       # vLLM 0.5 integration + experiment runner
│   ├── workload_generator.py   # Synthetic Poisson workload generator
│   └── metrics_collector.py    # TTFT, TPOT, QoE, SLO computation
├── run_all_experiments.py      # Entry point (--scheduler, --num-requests)
├── analyze_results.py          # 8 figures + summary CSV
├── config/experiment_config.yaml
├── results/1000/               # JSON results + summary.csv
├── figures/1000/               # 8 PNG figures
├── REPORT.md                   # Detailed experimental report
└── README.md                   # This file
```

---

## Figures (`figures/1000/`)

| File | Description |
|------|-------------|
| `fig_slo_attainment.png` | SLO Combined % — PROSPECT 2× baselines |
| `fig_ttft_comparison.png` | TTFT Mean + P99 side-by-side bars |
| `fig_ttft_cdf.png` | TTFT CDF — PROSPECT's bimodal distribution |
| `fig_ttft_p50.png` | Median TTFT — PROSPECT 3× lower |
| `fig_tpot_cdf.png` | TPOT CDF — all schedulers comparable |
| `fig_phase_breakdown.png` | Reasoning vs answering tokens per scheduler |
| `fig_slo_violation.png` | SLO violation rate (lower = better) |
| `fig_combined_slo_ttft.png` | Dual-axis: SLO% + P50 TTFT |

---

## Limitations

1. **Offline simulation** — vLLM batch mode, no live preemption between requests.
2. **Single GPU** — no multi-instance scheduling; PROSPECT's instance-selection absent.
3. **Synthetic workload** — not ShareGPT/LMSYS; real SHORT/LONG ratios may differ.
4. **max_new_tokens=1024** — real DeepSeek-R1 chains can reach 5000–8000 tokens.
5. **30s SLO** — too tight for 1000-req queue at 1 req/s; higher load or tighter SLO would stress-test further.

---

## Next Steps

- Real traces: ShareGPT, LMSYS-Chat-1M
- Live vLLM server mode with true token-level preemption
- Larger models: DeepSeek-R1-Distill-Qwen-14B, 32B
- Higher load: 2–5 req/s; tighter SLO: 10s TTFT
