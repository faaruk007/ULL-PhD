"""
scheduler_fcfs.py — First-Come-First-Served scheduler.
No phase awareness. Admits requests in arrival order up to max_concurrent.
"""
from typing import List
from .scheduler_base import BaseScheduler, RequestState


class FCFSScheduler(BaseScheduler):
    name = "fcfs"

    def next_batch(self, max_concurrent: int) -> List[RequestState]:
        slots = max_concurrent - len(self.active)
        admitted = []
        while slots > 0 and self.pending:
            req = self.pending.pop(0)
            req.submit_time = __import__("time").time()
            self.active[req.request_id] = req
            admitted.append(req)
            slots -= 1
        return admitted
