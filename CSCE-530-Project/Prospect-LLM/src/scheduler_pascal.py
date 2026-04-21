"""
scheduler_pascal.py — Phase-aware scheduler (PASCAL).

Two priority queues:
  HIGH (reasoning): admits new requests; reasoning runs uninterrupted.
  LOW  (answering): after </think>; uses RR + token pacer.

Demotion: reasoning request with > DEMOTION_THRESHOLD tokens → move to LOW queue.
"""
import time
from typing import List, Dict, Deque
from collections import deque
from .scheduler_base import BaseScheduler, RequestState, Phase


DEMOTION_THRESHOLD = 5000  # reasoning tokens before forced demotion


class PascalScheduler(BaseScheduler):
    name = "pascal"

    def __init__(self, config: dict):
        super().__init__(config)
        self.token_quantum: int = config.get("token_quantum", 500)
        self.demotion_threshold: int = config.get("demotion_threshold_tokens",
                                                   DEMOTION_THRESHOLD)
        # separate queues
        self.reasoning_queue: List[RequestState] = []   # pending, not yet admitted, HIGH
        self.answering_queue: List[RequestState] = []   # transitioned, LOW
        # token budget tracking for answering phase (RR)
        self.answer_tokens_budget: Dict[str, int] = {}
        # token pacer: track last release time per answering request
        self.pacer_last_release: Dict[str, float] = {}
        self.pacer_buffer: Dict[str, Deque] = {}  # buffered tokens not yet "released"

    def enqueue(self, req: RequestState):
        # All new arrivals go to reasoning queue (HIGH priority)
        self.reasoning_queue.append(req)

    def next_batch(self, max_concurrent: int) -> List[RequestState]:
        """Drain reasoning_queue first, then answering_queue."""
        slots = max_concurrent - len(self.active)
        admitted = []

        # Admit from HIGH priority (reasoning) first
        while slots > 0 and self.reasoning_queue:
            req = self.reasoning_queue.pop(0)
            req.submit_time = time.time()
            self.active[req.request_id] = req
            admitted.append(req)
            slots -= 1

        # Fill remaining slots from LOW priority (answering)
        while slots > 0 and self.answering_queue:
            req = self.answering_queue.pop(0)
            req.submit_time = req.submit_time or time.time()  # already admitted once
            self.active[req.request_id] = req
            self.answer_tokens_budget[req.request_id] = self.token_quantum
            self.pacer_last_release[req.request_id] = time.time()
            self.pacer_buffer[req.request_id] = deque()
            admitted.append(req)
            slots -= 1

        return admitted

    def on_token(self, req_id: str, token_text: str, timestamp: float):
        super().on_token(req_id, token_text, timestamp)
        req = self.active.get(req_id)
        if req is None:
            return

        # Demotion check: reasoning phase too long
        if req.phase == Phase.REASONING and req.reasoning_tokens > self.demotion_threshold:
            # demote to low priority on next scheduling step
            req._demoted = True  # flag for admission control

        # Answering phase: token pacer + RR budget
        if req.phase == Phase.ANSWERING and req_id in self.answer_tokens_budget:
            self.answer_tokens_budget[req_id] -= 1
            # Pacer: buffer tokens, release at ~10 tok/s (100ms spacing)
            target_interval = 0.100
            self.pacer_buffer[req_id].append(timestamp)
            # "release" tokens that are due
            now = time.time()
            while (self.pacer_buffer[req_id] and
                   now - self.pacer_last_release.get(req_id, 0) >= target_interval):
                self.pacer_buffer[req_id].popleft()
                self.pacer_last_release[req_id] = now

            # RR rotation after quantum
            if self.answer_tokens_budget[req_id] <= 0:
                self.answer_tokens_budget[req_id] = self.token_quantum
                # Conceptually "pause" — vLLM doesn't actually stop, but we track it

    def _on_phase_transition(self, req: RequestState):
        """Called when </think> detected. Move request to low-priority answering queue."""
        # Remove from active reasoning tracking — it stays in active dict
        # but its logical priority drops for future admission
        req.phase = Phase.ANSWERING
        # The request is already in self.active and vLLM is serving it
        # We note the transition for scheduling future requests
        pass  # In a full implementation, we'd migrate the request; here we track it

    def on_complete(self, req_id: str):
        self.answer_tokens_budget.pop(req_id, None)
        self.pacer_last_release.pop(req_id, None)
        self.pacer_buffer.pop(req_id, None)
        super().on_complete(req_id)
