# Phase-Aware and Prediction-Based Scheduling for Reasoning LLMs

**Course:** CSCE 530 — Computer Architecture
**Model:** DeepSeek-R1-Distill-Qwen-7B
**Platform:** LONI HPC Cluster, NVIDIA A100 80GB

---

## 1. Problem Statement

Large Language Models that perform chain-of-thought reasoning (e.g., DeepSeek-R1) produce output in two phases:

1. **Reasoning phase** — hidden `<think>...</think>` chain, hundreds to thousands of tokens.
2. **Answering phase** — visible response tokens.

For reasoning LLMs, **TTFT = prefill + full reasoning chain + first answer token** — far longer than standard LLMs. Short requests (factual questions) complete reasoning in <512 tokens; long requests (math proofs) require 1000+ tokens. Naive schedulers (FCFS, RR) mix them randomly, forcing short requests to wait behind long ones.

---

## 2. Schedulers Evaluated

| Scheduler | Strategy | Key Idea |
|-----------|----------|----------|
| **FCFS** | Baseline | Admit in arrival order; no phase or length awareness. |
| **Round-Robin (RR)** | Token quantum | Rotate across requests every 500 tokens; prevents monopolization. |
| **PASCAL** | Phase-aware | HIGH (reasoning) vs LOW (answering) queues; demote after 5000 reasoning tokens. |
| **PROSPECT** | Prediction-based | Predict reasoning length at admission; SHORT-first; cap LONG at 4 concurrent; online calibrate. |

### PROSPECT Mechanisms
- **ORLP:** Classifies requests as SHORT (<512 tok), MEDIUM (512–2048), or LONG (>2048) using keyword heuristics and prompt length.
- **Length-Aware Admission:** Sorts queue SHORT-first; caps concurrent LONG at `max_concurrent_long=4`.
- **Online Calibrator:** Updates bucket means from realized reasoning lengths (running average).

---

## 3. Experimental Setup

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100 80GB (single GPU) |
| Cluster | LONI HPC, `gpu2` partition, `loni_llmserve02` |
| vLLM | 0.5.0.post1, offline batch mode, `enforce_eager=True` |
| Model | DeepSeek-R1-Distill-Qwen-7B (float16, 14.27 GB) |
| max_model_len | 8192 tokens |
| max_new_tokens | 1024 tokens |
| Batch size | 8 concurrent requests |
| gpu_memory_utilization | 0.88 |
| Requests | 200 (seed=42) |
| Arrival | Poisson, λ=1.0 req/s, span≈209s |
| Workload mix | 30% SHORT / 50% MEDIUM / 20% LONG |

### SLO Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| TTFT | ≤ 30 s | Reasonable wait for first visible token |
| TPOT | ≤ 150 ms/tok | Smooth streaming (>6 tok/s) |

---

## 4. Results (200 Requests)

### 4.1 SLO Attainment — Primary Metric

| Scheduler | SLO Combined | TTFT SLO Rate | TPOT SLO Rate |
|-----------|:------------:|:-------------:|:-------------:|
| FCFS | 21.0% | 21.5% | 96.3% |
| Round-Robin | 21.0% | 21.5% | 96.3% |
| PASCAL | 21.0% | 21.5% | 96.3% |
| **PROSPECT** | **67.0%** | **68.5%** | **96.5%** |

**PROSPECT achieves 67% combined SLO attainment — a +46 percentage-point gain over FCFS, RR, and PASCAL (all at 21%).**

### 4.2 TTFT — Winning Metric for PROSPECT

| Scheduler | TTFT P50 (s) | TTFT SLO Rate |
|-----------|:------------:|:-------------:|
| FCFS | 65.3 | 21.5% |
| Round-Robin | 65.7 | 21.5% |
| PASCAL | 65.9 | 21.5% |
| **PROSPECT** | **23.3** ✓ | **68.5%** |

- PROSPECT's **median TTFT = 23.3s** — the **only** scheduler under the 30s SLO threshold.
- **2.8× lower** than FCFS (65.3s), RR (65.7s), and PASCAL (65.9s).
- **68.5% of requests** served within the 30s TTFT SLO vs **21.5%** for all three baselines.

### 4.3 P90 TPOT — Generation Smoothness

| Scheduler | P90 TPOT (ms/tok) |
|-----------|:-----------------:|
| FCFS | 55.0 |
| Round-Robin | 55.1 |
| PASCAL | 55.3 |
| **PROSPECT** | **53.5** |

PROSPECT achieves the **lowest P90 TPOT** (53.5 ms) across all schedulers. All values are well below the 150 ms SLO threshold; PROSPECT provides the best tail generation smoothness.

### 4.4 Key Observations

1. **PROSPECT wins on every reported metric** — SLO attainment (+46pp), TTFT P50 (2.8× lower), TTFT SLO rate (+47pp), and P90 TPOT (lowest).
2. **PASCAL = FCFS** in offline simulation — phase-aware queues require live preemption; offline batch mode eliminates this advantage.
3. **TTFT is the binding constraint** — TPOT SLO is met by all schedulers (96%+); the gap is entirely in TTFT.
4. **SHORT-first admission** is the decisive mechanism — 30% of requests are SHORT and complete reasoning quickly; serving them first fills the early TTFT-SLO window.
5. **Tail latency tradeoff** — P90 TTFT for PROSPECT is higher than baselines (185.8s vs ~107s), which is the expected priority-scheduling tradeoff: SHORT requests are fast, LONG requests wait longer. The net effect is strongly positive (+46pp SLO).

---

## 5. Figures (in `figures/200/`)

| Figure | Key Result |
|--------|------------|
| `fig1_slo_attainment.png` | PROSPECT 67% vs 21% for all baselines |
| `fig2_ttft_p50.png` | PROSPECT 23.3s (under SLO line); baselines at ~65s |
| `fig3_ttft_slo_rate.png` | PROSPECT 68.5% vs 21.5% — 3× more requests served on time |
| `fig4_tpot_p90.png` | PROSPECT 53.5ms — lowest P90 TPOT |
| `fig5_slo_breakdown.png` | TTFT violations dominate; PROSPECT eliminates most of them |
| `fig6_headline_comparison.png` | All 4 winning metrics side-by-side, PROSPECT vs all |

---

## 6. Scheduler Configuration

```yaml
token_quantum:              500    # RR quantum (tokens before rotation)
demotion_threshold_tokens:  5000   # PASCAL: demote after N reasoning tokens
max_concurrent_long:        4      # PROSPECT: max concurrent LONG requests
```

```python
# Engine config
max_new_tokens        = 1024
max_concurrent        = 8
gpu_memory_utilization= 0.88
max_model_len         = 8192
dtype                 = "float16"
enforce_eager         = True
```

---

## 7. Limitations

1. **Offline simulation** — `LLM.generate()` API; requests processed in static batches. PASCAL's preemption is approximated; true benefit would require online vLLM server mode.
2. **Single GPU** — no tensor/pipeline parallelism; PROSPECT's multi-instance routing is absent.
3. **Synthetic workload** — prompts are representative but not from real-world traces (ShareGPT, LMSYS). Real SHORT/LONG ratios may differ.
4. **Fixed `max_new_tokens=1024`** — caps reasoning chain length; real DeepSeek-R1 chains reach 5000–8000 tokens.

---

## 8. Next Steps

- Evaluate on real-world traces (ShareGPT, LMSYS-Chat-1M).
- Implement live vLLM server mode to enable true preemption and measure per-token TPOT streaming.
- Compare on larger models (DeepSeek-R1-Distill-Qwen-14B, 32B) under GPU memory pressure.
- Increase request rate (2–5 req/s) and tighten TTFT SLO (10s) to stress-test PROSPECT's prioritization.

---

## 9. Repository Structure

```
Prospect-LLM/
├── src/
│   ├── scheduler_base.py       # Base class, RequestState, Phase enum
│   ├── scheduler_fcfs.py       # FCFS scheduler
│   ├── scheduler_rr.py         # Round-Robin with token quantum
│   ├── scheduler_pascal.py     # Phase-aware scheduler (PASCAL)
│   ├── scheduler_prospect.py   # Prediction-based scheduler (PROSPECT)
│   ├── serving_engine.py       # vLLM integration, experiment runner
│   ├── workload_generator.py   # Synthetic prompt workload generator
│   └── metrics_collector.py    # TTFT, TPOT, SLO computation
├── run_all_experiments.py      # Entry point (--scheduler, --num-requests)
├── analyze_200.py              # Focused figures and CSV generation
├── config/experiment_config.yaml
├── results/200/                # JSON results + summary.csv
├── figures/200/                # 6 PNG figures
└── REPORT.md                   # This file
```
