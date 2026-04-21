"""
serving_engine.py — vLLM integration layer.

Wraps vLLM's LLM engine (offline / streaming mode).
Controls request admission based on scheduler decisions.
Streams output tokens to the scheduler's on_token() callback.
"""
import os
import sys
import time
import asyncio
import logging
from typing import List, Optional, Type
from pathlib import Path

from .scheduler_base import BaseScheduler, RequestState, Phase
from .workload_generator import generate_workload, save_workload, load_workload
from .metrics_collector import compute_metrics, save_results, print_summary

logger = logging.getLogger(__name__)


def run_experiment(
    scheduler_cls: Type[BaseScheduler],
    scheduler_config: dict,
    model_name: str,
    workload_path: Optional[str],
    num_requests: int,
    arrival_rate: float,
    max_new_tokens: int,
    max_concurrent: int,
    results_dir: str,
    seed: int = 42,
    gpu_memory_utilization: float = 0.88,
    max_model_len: int = 8192,
    dtype: str = "float16",
) -> dict:
    """
    Main experiment runner. Uses vLLM in synchronous offline mode.
    Simulates concurrent scheduling by processing requests in arrival-order
    batches and tracking per-request timing.
    """
    import vllm
    from vllm import LLM, SamplingParams

    print(f"\n{'*'*60}", flush=True)
    print(f"  Scheduler: {scheduler_cls.name.upper()}", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Requests: {num_requests}  |  Concurrency: {max_concurrent}", flush=True)
    print(f"{'*'*60}", flush=True)

    # Load or generate workload
    if workload_path and Path(workload_path).exists():
        requests = load_workload(workload_path)[:num_requests]
    else:
        requests = generate_workload(
            num_requests=num_requests,
            arrival_rate_rps=arrival_rate,
            seed=seed,
        )
        if workload_path:
            save_workload(requests, workload_path)

    # Initialize scheduler
    scheduler = scheduler_cls(scheduler_config)

    # Initialize vLLM engine (this loads the model — takes ~60s)
    print(f"  Loading model {model_name} ...", flush=True)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,   # skip CUDA graph compilation → faster startup
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        stop=None,
    )
    print("  Model loaded.", flush=True)

    # Convert relative arrival_times (from workload generator, Poisson-based, starting at 0)
    # to absolute wall-clock times anchored to when the experiment begins.
    wall_start = time.time()
    for req in requests:
        req.arrival_time += wall_start

    # Enqueue all requests
    for req in requests:
        scheduler.enqueue(req)

    # --- Main serving loop ---
    # We process requests in batches. Each batch is submitted to vLLM
    # (which handles actual GPU execution). We track timing per token
    # using streaming output.
    #
    # Note: vLLM 0.5.0 offline mode doesn't support true streaming per token,
    # but we can approximate per-token timing by measuring the total generation
    # time and interpolating, or by using the generate() call with token counting.
    #
    # We use a two-pass approach:
    #   Pass 1: run vLLM.generate() on the batch (gets all outputs)
    #   Pass 2: reconstruct per-token timestamps from the generation timestamps
    #           and phase boundaries detected in the output text.

    batch_num  = 0

    while scheduler.pending or scheduler.reasoning_queue or scheduler.answering_queue:
        # Get next batch to run
        batch = scheduler.next_batch(max_concurrent)
        if not batch:
            break

        batch_num += 1
        n_batch = len(batch)
        print(f"  Batch {batch_num}: {n_batch} requests", flush=True)

        # Submit to vLLM
        prompts = [req.prompt for req in batch]
        t_submit = time.time()

        try:
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        except Exception as e:
            logger.error(f"vLLM generate error: {e}")
            for req in batch:
                scheduler.on_complete(req.request_id)
            continue

        t_done = time.time()
        batch_elapsed = t_done - t_submit

        # Process outputs and reconstruct timing
        for req, output in zip(batch, outputs):
            generated_text = output.outputs[0].text if output.outputs else ""
            n_tokens = len(output.outputs[0].token_ids) if output.outputs else 0

            # Estimate per-token timing (uniform interpolation within batch)
            if n_tokens > 0:
                tok_interval = batch_elapsed / max(n_tokens, 1)
            else:
                tok_interval = 0.05  # fallback

            # Feed tokens to scheduler for phase tracking
            req.submit_time = t_submit
            t_current = t_submit

            # Detect </think> boundary position in token stream
            think_end_pos = None
            if "</think>" in generated_text:
                # Find approximately which token is the boundary
                partial = ""
                for i, tid in enumerate(output.outputs[0].token_ids):
                    # We don't have per-token text easily in offline mode
                    # Estimate: split generated_text proportionally
                    partial_frac = (i + 1) / n_tokens
                    partial_len  = int(len(generated_text) * partial_frac)
                    partial_text = generated_text[:partial_len]
                    if "</think>" in partial_text and think_end_pos is None:
                        think_end_pos = i
                        break

            # Feed tokens
            for i in range(n_tokens):
                t_tok = t_submit + i * tok_interval
                # Determine token text (approximation in offline mode)
                if i == 0:
                    tok_text = "<think>"
                elif think_end_pos is not None and i == think_end_pos:
                    tok_text = "</think>"
                else:
                    tok_text = "x"
                scheduler.on_token(req.request_id, tok_text, t_tok)

            # If </think> never detected in scheduler, do it from text
            if req.think_end_time is None and "</think>" in generated_text:
                frac = 0.5 if think_end_pos is None else think_end_pos / max(n_tokens, 1)
                req.think_end_time = t_submit + frac * batch_elapsed
                req.phase = Phase.ANSWERING
                # Reconstruct answering token times
                n_answer = n_tokens - (think_end_pos or int(n_tokens * 0.5))
                for j in range(n_answer):
                    req.answer_token_times.append(req.think_end_time + j * tok_interval)
                req.answering_tokens = n_answer
                req.reasoning_tokens = n_tokens - n_answer
            elif req.think_end_time is None:
                # No </think> — treat entire output as reasoning (unusual for DeepSeek-R1)
                req.reasoning_tokens = n_tokens
                req.think_end_time = t_done  # TTFT = full generation time

            req.total_output_tokens = n_tokens
            req.generated_text = generated_text
            req.completed = True
            scheduler.on_complete(req.request_id)

        print(f"    Done. Elapsed: {batch_elapsed:.1f}s", flush=True)

    wall_elapsed = time.time() - wall_start
    print(f"\n  Total wall time: {wall_elapsed:.1f}s", flush=True)

    # Collect all completed requests
    all_done = scheduler.done + list(scheduler.active.values())
    metrics = compute_metrics(all_done, scheduler_cls.name)
    metrics["wall_time_s"] = round(wall_elapsed, 2)
    metrics["batch_count"] = batch_num

    print_summary(metrics)

    # Save results
    out_path = Path(results_dir) / f"result_{scheduler_cls.name}.json"
    save_results(metrics, str(out_path))

    return metrics
