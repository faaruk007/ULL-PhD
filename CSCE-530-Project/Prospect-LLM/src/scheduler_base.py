"""
scheduler_base.py — shared request state, phase tracking, base scheduler class.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Phase(Enum):
    PREFILL   = "prefill"
    REASONING = "reasoning"   # inside <think>...</think>
    ANSWERING = "answering"   # after </think>
    DONE      = "done"


class ReasoningBucket(Enum):
    SHORT  = "short"    # < 512 reasoning tokens expected
    MEDIUM = "medium"   # 512 – 2048
    LONG   = "long"     # > 2048


THINK_END_STR = "</think>"


@dataclass
class RequestState:
    request_id:     str
    prompt:         str
    prompt_len:     int
    arrival_time:   float = field(default_factory=time.time)
    submit_time:    Optional[float] = None   # when admitted to engine
    first_token_time: Optional[float] = None # first ANY token (reasoning or answer)
    think_end_time: Optional[float] = None   # when </think> emitted = TTFT in PASCAL
    answer_token_times: List[float] = field(default_factory=list)
    total_output_tokens: int = 0
    reasoning_tokens: int = 0
    answering_tokens: int = 0
    phase: Phase = Phase.PREFILL
    reasoning_bucket: ReasoningBucket = ReasoningBucket.MEDIUM
    generated_text: str = ""
    completed: bool = False

    # token-pacer buffer (for answering phase)
    token_buffer: List[float] = field(default_factory=list)  # timestamps of buffered tokens

    @property
    def ttft(self) -> Optional[float]:
        """TTFT in PASCAL's definition: time from arrival to first ANSWERING token."""
        if self.think_end_time is not None:
            return self.think_end_time - self.arrival_time
        return None

    @property
    def tpot_ms(self) -> Optional[float]:
        """Mean time per output token (answering phase only), ms."""
        if len(self.answer_token_times) < 2:
            return None
        deltas = [self.answer_token_times[i+1] - self.answer_token_times[i]
                  for i in range(len(self.answer_token_times)-1)]
        return (sum(deltas) / len(deltas)) * 1000

    @property
    def qoe(self) -> Optional[float]:
        """Simplified QoE: ratio of tokens delivered on-time to expected tokens.
        We define on-time as TPOT ≤ target_tpot_ms.
        """
        if not self.answer_token_times or len(self.answer_token_times) < 2:
            return 1.0
        target_interval = 0.100  # 100ms = 10 tok/s
        on_time = 0
        for i in range(1, len(self.answer_token_times)):
            dt = self.answer_token_times[i] - self.answer_token_times[i-1]
            if dt <= target_interval * 1.5:  # 50% slack
                on_time += 1
        return on_time / (len(self.answer_token_times) - 1)


class BaseScheduler:
    """Abstract base class for all schedulers."""

    name: str = "base"

    def __init__(self, config: dict):
        self.config = config
        self.pending: List[RequestState] = []
        self.active:  Dict[str, RequestState] = {}
        self.done:    List[RequestState] = []
        # These are overridden by PascalScheduler/ProspectScheduler;
        # defined here so the while-loop condition in serving_engine works for all schedulers.
        self.reasoning_queue: List[RequestState] = []
        self.answering_queue: List[RequestState] = []

    def enqueue(self, req: RequestState):
        self.pending.append(req)

    def next_batch(self, max_concurrent: int) -> List[RequestState]:
        """Return requests to admit to the engine this step. Override per scheduler."""
        raise NotImplementedError

    def on_token(self, req_id: str, token_text: str, timestamp: float):
        """Called for every generated token. Update phase state."""
        req = self.active.get(req_id)
        if req is None:
            return

        req.total_output_tokens += 1
        req.generated_text += token_text

        if req.first_token_time is None:
            req.first_token_time = timestamp

        # Phase transition detection
        if req.phase in (Phase.PREFILL, Phase.REASONING):
            req.reasoning_tokens += 1
            req.phase = Phase.REASONING
            if THINK_END_STR in req.generated_text and req.think_end_time is None:
                req.think_end_time = timestamp
                req.phase = Phase.ANSWERING
                self._on_phase_transition(req)
        elif req.phase == Phase.ANSWERING:
            req.answering_tokens += 1
            req.answer_token_times.append(timestamp)

    def on_complete(self, req_id: str):
        req = self.active.pop(req_id, None)
        if req:
            req.completed = True
            req.phase = Phase.DONE
            self.done.append(req)

    def _on_phase_transition(self, req: RequestState):
        """Hook for subclasses to handle reasoning→answering transition."""
        pass
