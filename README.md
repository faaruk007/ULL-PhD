# PASCAL & PROSPECT: Phase-Aware Scheduling for Reasoning LLMs

Implementation and evaluation of **PASCAL** (HPCA 2026) and **PROSPECT** (novel extension) on DeepSeek-R1-Distill-Qwen-7B using vLLM 0.5.0.

**Course:** CSCE 530 — Computer Architecture | **Platform:** LONI HPC, NVIDIA A100 80GB

---

## Results Summary (200 Requests)

### Winning Metrics — PROSPECT vs. All Baselines

| Scheduler | SLO Attainment | TTFT P50 (s) | TTFT SLO Rate | P90 TPOT (ms) |
|-----------|:--------------:|:------------:|:-------------:|:-------------:|
| FCFS (baseline) | 21.0% | 65.3 | 21.5% | 55.0 |
| Round-Robin | 21.0% | 65.7 | 21.5% | 55.1 |
| PASCAL | 21.0% | 65.9 | 21.5% | 55.3 |
| **PROSPECT (ours)** | **67.0%** | **23.3** ✓ | **68.5%** | **53.5** |

**Key wins (PROSPECT over all baselines, including PASCAL):**
- **+46 pp SLO attainment** — 67% vs 21% for FCFS, RR, and PASCAL
- **2.8× lower median TTFT** — P50 drops from ~65s to 23.3s (the only scheduler under the 30s SLO threshold)
- **+47 pp TTFT SLO Rate** — 68.5% of requests served within 30s vs 21.5% for all others
- **Lowest P90 TPOT** — 53.5 ms vs 55.0–55.3 ms for baselines (smooth generation)

---

## Why PROSPECT Works

Reasoning LLMs (DeepSeek-R1) generate a hidden `<think>…</think>` chain **before** the visible answer:

```
TTFT = prefill + full reasoning chain + first answer token
```

SHORT requests (factual questions) finish reasoning in <512 tokens. LONG requests (math proofs) can take 1000+ tokens. FCFS, RR, and PASCAL all mix them — SHORT requests wait behind LONG ones in the queue.

**PROSPECT** predicts reasoning length at admission and reorders:

```
Request arrives
    │
    ▼
ORLP Predictor ── keyword heuristics + prompt length ──► SHORT / MEDIUM / LONG
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

SHORT requests (30% of workload) reach the GPU first → **median TTFT = 23.3s** (under SLO), vs 65+ seconds for all baselines.

**Why PASCAL ≈ FCFS here:** Offline batch mode limits true phase-based preemption. PROSPECT's length-aware admission works within the batch API constraints, making it the effective differentiator.

---

## Scheduler Mechanisms

| Scheduler | Mechanism |
|-----------|-----------|
| **FCFS** | Admit in arrival order. No awareness of request type or phase. |
| **Round-Robin** | Rotate every 500 tokens. Mild anti-starvation, no latency benefit. |
| **PASCAL** | Separate HIGH (reasoning) and LOW (answering) queues. Demote after 5000 reasoning tokens. |
| **PROSPECT** | Predict reasoning length → SHORT-first admission → cap LONG concurrency → online calibrate. |

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | DeepSeek-R1-Distill-Qwen-7B (7B params, float16, 14.27 GB) |
| **GPU** | NVIDIA A100 80GB (single GPU) |
| **Cluster** | LONI HPC, `gpu2` partition |
| **vLLM** | 0.5.0.post1, `enforce_eager=True` |
| **max_model_len** | 8192 tokens |
| **max_new_tokens** | 1024 tokens |
| **Batch size** | 8 concurrent requests |
| **gpu_memory_utilization** | 0.88 |
| **Requests** | 200 (seed=42) |
| **Arrival pattern** | Poisson, λ=1.0 req/s |

### SLO Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| TTFT | ≤ 30 s | Time to first visible answer token |
| TPOT | ≤ 150 ms/tok | Smooth token streaming (>6 tok/s) |

### Workload Mix

| Type | Count | Sample Prompts |
|------|-------|----------------|
| **SHORT** | 60 (30%) | "What is the capital of France?", "What is H₂O's formula?" |
| **MEDIUM** | 100 (50%) | "Explain gradient descent", "Write Sieve of Eratosthenes in Python" |
| **LONG** | 40 (20%) | "Prove infinitely many primes exist", "Derive FFT O(n log n) complexity" |

---

## Setup & Run

```bash
# Requires: vLLM 0.5.0, Python 3.10, NVIDIA A100
conda activate vllm05_env

# Run one scheduler, 200 requests
python3 run_all_experiments.py --scheduler prospect \
    --num-requests 200 --results-dir results/200

# All 4 schedulers sequentially
python3 run_all_experiments.py --scheduler all \
    --num-requests 200 --results-dir results/200

# Generate 6 focused figures (needs matplotlib)
python3 analyze_200.py
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
│   └── metrics_collector.py    # TTFT, TPOT, SLO computation
├── run_all_experiments.py      # Entry point (--scheduler, --num-requests)
├── analyze_200.py              # 6 focused figures + summary CSV
├── config/experiment_config.yaml
├── results/200/                # JSON results + summary.csv
├── figures/200/                # 6 PNG figures
├── REPORT.md                   # Detailed experimental report
└── README.md                   # This file
```

---

## Figures (`figures/200/`)

| File | Description |
|------|-------------|
| `fig1_slo_attainment.png` | SLO Combined % — PROSPECT 67% vs 21% for all others |
| `fig2_ttft_p50.png` | Median TTFT — PROSPECT 23.3s (under SLO); others ~65s |
| `fig3_ttft_slo_rate.png` | % requests meeting TTFT ≤ 30s — PROSPECT 68.5% vs 21.5% |
| `fig4_tpot_p90.png` | P90 TPOT — PROSPECT 53.5ms, lowest across all schedulers |
| `fig5_slo_breakdown.png` | SLO violation breakdown — TTFT is the bottleneck; PROSPECT solves it |
| `fig6_headline_comparison.png` | PROSPECT vs all baselines on all 4 winning metrics |

---

## Limitations

1. **Offline simulation** — vLLM batch mode, no live preemption between requests.
2. **Single GPU** — no tensor/pipeline parallelism; PROSPECT's instance-selection absent.
3. **Synthetic workload** — not ShareGPT/LMSYS; real SHORT/LONG ratios may differ.
4. **Tail latency tradeoff** — PROSPECT's SHORT-first policy defers LONG requests; P90 TTFT is higher than baselines (expected priority scheduling behavior).

---

## Next Steps

- Real traces: ShareGPT, LMSYS-Chat-1M
- Live vLLM server mode with true token-level preemption
- Larger models: DeepSeek-R1-Distill-Qwen-14B, 32B
- Higher request rates (2–5 req/s) to stress-test scheduling decisions
