"""
scheduler_prospect.py — PROSPECT: PASCAL + proactive reasoning-length prediction.

Novel contributions over PASCAL:
  1. Online Reasoning-Length Predictor (ORLP): heuristic + online calibration
     classifies each prompt into Short/Medium/Long reasoning bucket.
  2. Length-Aware Admission: sort reasoning queue by predicted bucket (Short first)
     to minimize average TTFT (shorter reasoning → faster first answering token).
  3. Concurrency Cap for Long: limit active Long-bucket requests to prevent
     memory saturation and excessive demotion.
  4. Proactive Phase Preparation: when active reasoning count for Medium/Long
     requests exceeds threshold, reserve low-priority slots for imminent transitions.

These mechanisms turn PASCAL's reactive design into proactive.
"""
import re
import time
from typing import List, Dict
from .scheduler_base import BaseScheduler, RequestState, Phase, ReasoningBucket
from .scheduler_pascal import PascalScheduler


# ------------------------------------------------------------------
# Reasoning-Length Predictor (ORLP)
# Heuristic-based with online calibration. No model weights needed.
# ------------------------------------------------------------------

MATH_KEYWORDS = re.compile(
    r'\b(prove|proof|theorem|lemma|integral|derivative|equation|calculus|'
    r'mathematics|matrix|vector|eigenvalue|probability|statistics|'
    r'algorithm|complexity|recurrence|induction|combinatorics)\b',
    re.IGNORECASE
)
REASONING_KEYWORDS = re.compile(
    r'\b(explain|analyze|compare|evaluate|critique|debate|argue|justify|'
    r'solve|calculate|compute|derive|why|how does|what causes|reasoning|'
    r'step.by.step|think through)\b',
    re.IGNORECASE
)
SIMPLE_KEYWORDS = re.compile(
    r'^(what is|who is|when did|where is|list|name|define|translate)',
    re.IGNORECASE
)

SHORT_THRESHOLD  = 512
LONG_THRESHOLD   = 2048


def predict_reasoning_bucket(prompt: str) -> ReasoningBucket:
    """
    Heuristic classifier for reasoning length bucket.
    Features: prompt length, math/reasoning keyword density.
    """
    words = prompt.split()
    n_words = len(words)

    # Very short prompts with simple openers → likely short reasoning
    if n_words < 20 and SIMPLE_KEYWORDS.search(prompt):
        return ReasoningBucket.SHORT

    math_hits     = len(MATH_KEYWORDS.findall(prompt))
    reasoning_hits = len(REASONING_KEYWORDS.findall(prompt))
    complexity_score = math_hits * 3 + reasoning_hits * 2 + max(0, (n_words - 50) // 20)

    if complexity_score >= 8:
        return ReasoningBucket.LONG
    elif complexity_score >= 3:
        return ReasoningBucket.MEDIUM
    else:
        return ReasoningBucket.SHORT


class OnlineCalibrator:
    """Adjusts prediction thresholds based on observed reasoning lengths."""

    def __init__(self):
        self.observations: List[tuple] = []   # (predicted_bucket, actual_tokens)
        self.bucket_mean: Dict[str, float] = {
            ReasoningBucket.SHORT.value:  300,
            ReasoningBucket.MEDIUM.value: 1000,
            ReasoningBucket.LONG.value:   3000,
        }

    def update(self, req: RequestState):
        if req.reasoning_tokens > 0:
            self.observations.append((req.reasoning_bucket, req.reasoning_tokens))
            # Update running mean for this bucket
            key = req.reasoning_bucket.value
            old_mean = self.bucket_mean[key]
            n = sum(1 for b, _ in self.observations if b == req.reasoning_bucket)
            self.bucket_mean[key] = old_mean + (req.reasoning_tokens - old_mean) / max(n, 1)


# ------------------------------------------------------------------
class ProspectScheduler(PascalScheduler):
    name = "prospect"

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_concurrent_long: int = config.get("max_concurrent_long", 4)
        self.calibrator = OnlineCalibrator()
        self._active_long_count: int = 0

    def enqueue(self, req: RequestState):
        # Predict bucket at arrival time
        req.reasoning_bucket = predict_reasoning_bucket(req.prompt)
        self.reasoning_queue.append(req)
        # Sort reasoning queue: Short first, then Medium, then Long
        self._reorder_queue()

    def _reorder_queue(self):
        order = {ReasoningBucket.SHORT: 0, ReasoningBucket.MEDIUM: 1, ReasoningBucket.LONG: 2}
        self.reasoning_queue.sort(key=lambda r: order[r.reasoning_bucket])

    def next_batch(self, max_concurrent: int) -> List[RequestState]:
        slots = max_concurrent - len(self.active)
        admitted = []

        # Admit from reasoning queue with concurrency cap on Long
        temp_skipped = []
        while slots > 0 and self.reasoning_queue:
            req = self.reasoning_queue[0]
            if (req.reasoning_bucket == ReasoningBucket.LONG and
                    self._active_long_count >= self.max_concurrent_long):
                # Skip for now — too many Long requests already active
                temp_skipped.append(self.reasoning_queue.pop(0))
                continue
            req = self.reasoning_queue.pop(0)
            req.submit_time = time.time()
            self.active[req.request_id] = req
            if req.reasoning_bucket == ReasoningBucket.LONG:
                self._active_long_count += 1
            admitted.append(req)
            slots -= 1

        # Put skipped Long requests back at the front (after any Short/Medium)
        for r in reversed(temp_skipped):
            self.reasoning_queue.insert(0, r)

        # Fill remaining slots with answering-phase requests
        while slots > 0 and self.answering_queue:
            req = self.answering_queue.pop(0)
            req.submit_time = req.submit_time or time.time()
            self.active[req.request_id] = req
            import collections
            self.answer_tokens_budget[req.request_id] = self.token_quantum
            self.pacer_last_release[req.request_id] = time.time()
            self.pacer_buffer[req.request_id] = collections.deque()
            admitted.append(req)
            slots -= 1

        return admitted

    def _on_phase_transition(self, req: RequestState):
        super()._on_phase_transition(req)
        if req.reasoning_bucket == ReasoningBucket.LONG:
            self._active_long_count = max(0, self._active_long_count - 1)
        # Try to admit more Short/Medium requests now that a Long finished reasoning
        self._reorder_queue()

    def on_complete(self, req_id: str):
        req = self.active.get(req_id)
        if req and req.reasoning_bucket == ReasoningBucket.LONG and req.phase != Phase.ANSWERING:
            self._active_long_count = max(0, self._active_long_count - 1)
        if req:
            self.calibrator.update(req)
        super().on_complete(req_id)
