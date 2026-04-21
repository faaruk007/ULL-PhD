"""
scheduler_rr.py — Round-Robin scheduler with token quantum.
Rotates among active requests; after each quantum, the request is paused
(moved to back of round queue) and the next one gets a turn.
In practice, since vLLM handles the actual generation, we simulate RR by
interleaving concurrent requests and tracking per-request token budgets.
"""
import time
from typing import List, Dict
from .scheduler_base import BaseScheduler, RequestState


class RoundRobinScheduler(BaseScheduler):
    name = "rr"

    def __init__(self, config: dict):
        super().__init__(config)
        self.token_quantum: int = config.get("token_quantum", 500)
        self.tokens_this_quantum: Dict[str, int] = {}  # request_id -> tokens this turn
        self.rr_queue: List[str] = []  # ordered list of active request IDs

    def next_batch(self, max_concurrent: int) -> List[RequestState]:
        slots = max_concurrent - len(self.active)
        admitted = []
        while slots > 0 and self.pending:
            req = self.pending.pop(0)
            req.submit_time = time.time()
            self.active[req.request_id] = req
            self.rr_queue.append(req.request_id)
            self.tokens_this_quantum[req.request_id] = 0
            admitted.append(req)
            slots -= 1
        return admitted

    def on_token(self, req_id: str, token_text: str, timestamp: float):
        super().on_token(req_id, token_text, timestamp)
        if req_id in self.tokens_this_quantum:
            self.tokens_this_quantum[req_id] += 1
            # After quantum exhausted, rotate (move to back of rr_queue)
            if self.tokens_this_quantum[req_id] >= self.token_quantum:
                self.tokens_this_quantum[req_id] = 0
                if req_id in self.rr_queue:
                    self.rr_queue.remove(req_id)
                    self.rr_queue.append(req_id)  # move to back

    def on_complete(self, req_id: str):
        if req_id in self.rr_queue:
            self.rr_queue.remove(req_id)
        self.tokens_this_quantum.pop(req_id, None)
        super().on_complete(req_id)
