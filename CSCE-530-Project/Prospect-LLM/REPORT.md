# Phase-Aware and Prediction-Based Scheduling for Reasoning LLMs

**Course:** CSCE 530 — Computer Architecture
**Model:** DeepSeek-R1-Distill-Qwen-7B
**Platform:** LONI HPC Cluster, NVIDIA A100 80GB

---

## 1. Problem Statement

Large Language Models (LLMs) that perform chain-of-thought reasoning (e.g., DeepSeek-R1) produce output in two distinct phases:

1. **Reasoning phase** — the model generates a hidden `<think>...</think>` chain before producing a visible answer. This can span hundreds to thousands of tokens.
2. **Answering phase** — the model emits the final response token by token.

Traditional schedulers (FCFS, Round-Robin) treat all tokens uniformly. They are unaware that:
- **TTFT (Time-to-First-Token)** for reasoning LLMs includes the full reasoning chain, not just the prefill phase.
- Short requests (simple factual questions) have very short reasoning chains, while complex math/proof requests can have very long ones.
- Mixing SHORT and LONG requests in the same batch without awareness of reasoning length causes SHORT requests to wait unnecessarily, wasting their low-latency potential.

---

## 2. Schedulers Evaluated

| Scheduler | Strategy | Key Idea |
|-----------|----------|----------|
| **FCFS** | Baseline | Admit requests in arrival order. No phase awareness. |
| **Round-Robin (RR)** | Token quantum | Rotate across requests every 500 tokens. Prevents monopolization. |
| **PASCAL** | Phase-aware | Separate HIGH (reasoning) and LOW (answering) priority queues. Demote requests exceeding 5000 reasoning tokens. |
| **PROSPECT** | Prediction-based | Predict reasoning length at admission. Prioritize SHORT requests; cap concurrent LONG requests at 4. Online calibrator updates bucket means. |

### PROSPECT Key Mechanisms
- **Online Reasoning-Length Predictor (ORLP):** Classifies each request as SHORT (<512 tokens), MEDIUM (512–2048), or LONG (>2048) using keyword heuristics and prompt length.
- **Length-Aware Admission:** Sorts reasoning queue SHORT-first; limits concurrent LONG requests (`max_concurrent_long=4`).
- **Online Calibrator:** Tracks realized vs. predicted reasoning lengths; updates bucket statistics via running average.

---

## 3. Experimental Setup

### Hardware
- **GPU:** NVIDIA A100 80GB (single GPU)
- **Cluster:** LONI HPC, `gpu2` partition, account `loni_llmserve02`
- **vLLM Version:** 0.5.0.post1 (offline batch mode, `enforce_eager=True`)

### Model
- **DeepSeek-R1-Distill-Qwen-7B** — 7B parameter reasoning model distilled from DeepSeek-R1 (671B). Based on Qwen2-7B architecture. Generates native `<think>...</think>` reasoning traces. Available at `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` on HuggingFace.
- **Precision:** float16 (cast from bfloat16), `max_model_len=8192`
- **Weight size:** 14.27 GB

### Workload (200-request run)
- **Total requests:** 200 (seed=42, Poisson inter-arrival at 1.0 req/s, arrival span ≈209s)
- **Mix:** 30% SHORT (simple factual), 50% MEDIUM (explanation/coding), 20% LONG (math/proofs)
- **max_new_tokens:** 1024 | **max_concurrent:** 8 (batch size)
- **Same workload trace** reused across all 4 schedulers for fair comparison

### Sample Prompts
- **SHORT:** "What is the capital of France?", "What is the chemical formula for water?"
- **MEDIUM:** "Explain how gradient descent works", "Write a Python function for Sieve of Eratosthenes"
- **LONG:** "Prove there are infinitely many primes", "Derive backpropagation equations for a two-layer neural network"

### SLO Thresholds
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| TTFT | ≤ 30 seconds | Reasonable wait for first visible answer |
| TPOT | ≤ 150 ms/token | Smooth streaming experience |
| QoE | ≥ 0.95 | Normalized token delivery quality |

---

## 4. Results (1000 Requests)

### SLO Attainment

| Scheduler | SLO Combined | TTFT SLO% | TPOT SLO% | QoE SLO% |
|-----------|-------------|-----------|-----------|---------|
| FCFS | 4.6% | 4.8% | 96.4% | 98.4% |
| Round-Robin | 5.1% | 5.3% | 96.4% | 98.4% |
| PASCAL | 5.1% | 5.3% | 96.4% | 98.4% |
| **PROSPECT** | **10.6%** | **11.0%** | **96.5%** | **98.4%** |

### Latency Metrics

| Scheduler | TTFT Mean (s) | TTFT P50 (s) | TTFT P90 (s) | TTFT P99 (s) | TPOT Mean (ms) |
|-----------|--------------|-------------|-------------|-------------|---------------|
| FCFS | 312.7 | 318.5 | 576.1 | 622.5 | 31.9 |
| Round-Robin | 291.7 | 297.3 | 538.0 | 580.4 | 31.1 |
| PASCAL | 289.7 | 295.2 | 534.4 | 576.9 | 31.0 |
| **PROSPECT** | 293.6 | **103.6** | 934.4 | 1355.9 | **30.5** |

### Token Statistics

| Scheduler | Reasoning Tokens (mean) | Answering Tokens (mean) | Total Tokens |
|-----------|------------------------|------------------------|-------------|
| FCFS | 715.4 | 110.5 | 825,921 |
| Round-Robin | 715.4 | 110.5 | 825,921 |
| PASCAL | 715.4 | 110.5 | 825,921 |
| PROSPECT | 712.7 | 114.4 | 827,078 |

### Key Observations
1. **PROSPECT achieves +6pp SLO gain** over FCFS and **3× lower median TTFT** (P50: 103.6s vs 318.5s) via SHORT-first admission control.
2. **TTFT is the bottleneck** — TPOT and QoE are nearly identical across all schedulers; vLLM's batching handles streaming uniformly.
3. **PROSPECT tail latency is high** (P90=934s, P99=1356s) — LONG requests are deferred to serve SHORT ones first. This is the inherent fairness tradeoff of priority scheduling.
4. **PASCAL ≈ FCFS** in offline simulation — phase-aware queues require live preemption to show benefit; the offline batch model limits this.
5. **Low absolute SLO%** is expected: 1000 requests at 1 req/s spans 1031s; with 125 batches × 13s, only requests admitted in the first ~2 batches can achieve TTFT ≤ 30s regardless of scheduler.

---

## 5. Figures

All figures saved in `figures/1000/`:

| Figure | Description |
|--------|-------------|
| `fig_slo_attainment.png` | SLO Combined % — bar chart |
| `fig_ttft_comparison.png` | TTFT Mean + P99 side-by-side |
| `fig_ttft_cdf.png` | CDF of TTFT — bimodal shape for PROSPECT |
| `fig_ttft_p50.png` | Median TTFT — PROSPECT 3× lower |
| `fig_tpot_cdf.png` | CDF of TPOT — all schedulers comparable |
| `fig_phase_breakdown.png` | Reasoning vs answering tokens |
| `fig_slo_violation.png` | Violation rate (lower = better) |
| `fig_combined_slo_ttft.png` | Dual-axis: SLO% + P50 TTFT |

---

## 6. Scheduler Configuration

```yaml
token_quantum:              500    # RR quantum (tokens before rotation)
demotion_threshold_tokens:  5000   # PASCAL: demote after N reasoning tokens
max_concurrent_long:        4      # PROSPECT: max concurrent LONG requests
```

```python
# Engine config (1000-req run)
max_new_tokens        = 1024
max_concurrent        = 8
gpu_memory_utilization= 0.88
max_model_len         = 8192
dtype                 = "float16"
enforce_eager         = True      # skip CUDA graph compilation
```

---

## 7. Limitations and Discussion

1. **Offline simulation** — we use vLLM's offline `LLM.generate()` API. Requests are processed in static batches of 8, not a true continuous-batching live server. PASCAL's preemption/migration mechanisms are approximated.
2. **Single GPU** — no multi-instance parallelism (tensor parallelism or pipeline parallelism). PROSPECT's instance-selection component is absent.
3. **Synthetic workload** — prompts are representative but not from a real-world distribution (e.g., ShareGPT, LMSYS). Real workloads may have different SHORT/LONG ratios.
4. **SLO thresholds** — the 30s TTFT SLO is chosen conservatively for reasoning LLMs; a tighter SLO (e.g., 10s) would show starker differences.
5. **Fixed `max_new_tokens=1024`** — caps reasoning chain length; real DeepSeek-R1 chains can be 5000–8000 tokens.

---

## 8. Next Steps

- Evaluate on **real-world traces** (ShareGPT, LMSYS-Chat-1M) with actual reasoning length distributions.
- Implement a **live vLLM server mode** (online API) to enable true preemption and measure streaming TPOT per token.
- Compare on **larger models** (DeepSeek-R1-Distill-Qwen-14B, 32B) to test GPU memory pressure effects.
- Tighten **TTFT SLO to 10s** to stress-test PROSPECT's SHORT-first prioritization.

---

## 9. Repository Structure

```
Experiment/
├── src/
│   ├── scheduler_base.py       # Base class, RequestState, Phase enum
│   ├── scheduler_fcfs.py       # FCFS scheduler
│   ├── scheduler_rr.py         # Round-Robin with token quantum
│   ├── scheduler_pascal.py     # Phase-aware scheduler (PASCAL)
│   ├── scheduler_prospect.py   # Prediction-based scheduler (PROSPECT)
│   ├── serving_engine.py       # vLLM integration, experiment runner
│   ├── workload_generator.py   # Synthetic prompt workload generator
│   └── metrics_collector.py    # TTFT, TPOT, QoE, SLO computation
├── run_all_experiments.py      # Entry point (--scheduler, --num-requests)
├── analyze_results.py          # Figure and CSV generation
├── config/experiment_config.yaml
├── results/                    # JSON results per scheduler
├── figures/                    # Generated PNG figures
└── REPORT.md                   # This file
```
