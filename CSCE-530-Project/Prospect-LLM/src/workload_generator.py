"""
workload_generator.py — Generate 200-request workload traces.

Produces a mix of:
  - Short-reasoning:  factual/simple queries (~30%)
  - Medium-reasoning: explanation/coding queries (~50%)
  - Long-reasoning:   math/proof/complex queries (~20%)

Prompts are real-world style so DeepSeek-R1 actually generates <think> tokens.
"""
import json
import random
import time
from pathlib import Path
from typing import List
from .scheduler_base import RequestState


# Representative prompts per category
SHORT_PROMPTS = [
    "What is the capital of France?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the boiling point of water in Celsius?",
    "Define photosynthesis in one sentence.",
    "What year did World War II end?",
    "Name the planets in our solar system.",
    "What is the speed of light in m/s?",
    "Who invented the telephone?",
    "What is the chemical formula for water?",
    "When was the Eiffel Tower built?",
    "What is the largest ocean on Earth?",
    "Who was the first president of the United States?",
    "What is 17 multiplied by 13?",
    "What language is spoken in Brazil?",
    "Define the term 'entropy' in one sentence.",
    "What is the square root of 144?",
    "Who painted the Mona Lisa?",
    "What is the atomic number of carbon?",
    "Name three programming languages.",
    "What does CPU stand for?",
]

MEDIUM_PROMPTS = [
    "Explain how gradient descent works in machine learning and why it converges.",
    "What are the key differences between TCP and UDP protocols? When would you use each?",
    "Explain the concept of virtual memory and how an operating system manages it.",
    "How does HTTPS ensure secure communication? Explain the TLS handshake process.",
    "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes. Explain your approach.",
    "Explain the CAP theorem in distributed systems and its practical implications.",
    "How does a neural network learn? Describe backpropagation step by step.",
    "Compare and contrast SQL and NoSQL databases. When would you choose each?",
    "Explain how Git's version control system works internally.",
    "What is the difference between concurrency and parallelism? Give examples.",
    "How does the PageRank algorithm work? Explain the key intuition.",
    "Describe the process of compiling a C program, from source code to executable.",
    "Explain what a hash table is and how collision resolution works.",
    "How does a transformer model process a sequence of tokens? Describe attention.",
    "What are the SOLID principles in object-oriented programming? Give examples.",
    "Explain the difference between dynamic programming and greedy algorithms.",
    "How does garbage collection work in Java? Describe the generational approach.",
    "What is the difference between process and thread? How does context switching work?",
    "Explain the MapReduce programming model and give an example use case.",
    "How does a B-tree differ from a binary search tree? Why is it used in databases?",
    "Describe how public-key cryptography works and give an example application.",
    "Explain the concept of load balancing and describe common strategies.",
    "How does Docker containerization work? What is the difference from a VM?",
    "Explain the bias-variance tradeoff in machine learning.",
    "What is the difference between supervised and unsupervised learning? Give examples.",
]

LONG_PROMPTS = [
    "Prove that there are infinitely many prime numbers using Euclid's proof. Then explain why this proof is considered elegant.",
    "Solve the following recurrence relation: T(n) = 2T(n/2) + n log n. Show all steps using the Master Theorem.",
    "Derive the backpropagation equations for a two-layer neural network with sigmoid activations. Show all partial derivatives explicitly.",
    "Explain why P ≠ NP is considered the most important open problem in computer science. Describe what a proof would imply.",
    "A database needs to handle 10,000 transactions per second with ACID guarantees. Design the architecture, explain your choices.",
    "Prove that the time complexity of the Fast Fourier Transform (FFT) is O(n log n). Explain the divide-and-conquer structure.",
    "Derive the formula for the expected number of comparisons in QuickSort. Show the full mathematical analysis.",
    "Explain the traveling salesman problem, prove it is NP-hard, and describe the best known approximation algorithms.",
    "Prove that matrix multiplication cannot be done in less than Ω(n²) time in the decision tree model.",
    "Design a distributed consensus algorithm from scratch. Prove it is correct under the conditions you specify.",
    "Analyze the convergence properties of stochastic gradient descent. Under what conditions does it converge?",
    "Explain and prove the correctness of Dijkstra's algorithm. Then extend it to handle negative weights.",
    "Derive the Fourier series representation of a square wave. Show all integration steps.",
    "Prove the correctness of the Union-Find data structure with path compression and union by rank.",
    "Design a memory allocator that minimizes fragmentation. Analyze the time and space complexity.",
    "Prove that any comparison-based sorting algorithm requires Ω(n log n) comparisons in the worst case.",
    "Explain and mathematically analyze the PageRank algorithm. Prove it converges for any graph.",
    "Derive the equations for linear regression using the method of least squares. Show the matrix form solution.",
    "Design and prove the correctness of a lock-free concurrent queue using compare-and-swap operations.",
    "Explain the relationship between entropy, information theory, and machine learning loss functions.",
]


def generate_workload(
    num_requests: int = 200,
    arrival_rate_rps: float = 1.0,
    seed: int = 42,
    short_frac: float = 0.30,
    medium_frac: float = 0.50,
    # long_frac is remainder = 0.20
) -> List[RequestState]:
    rng = random.Random(seed)

    n_short  = int(num_requests * short_frac)
    n_medium = int(num_requests * medium_frac)
    n_long   = num_requests - n_short - n_medium

    # Sample prompts with replacement if needed
    short_pool  = SHORT_PROMPTS * (n_short  // len(SHORT_PROMPTS)  + 2)
    medium_pool = MEDIUM_PROMPTS * (n_medium // len(MEDIUM_PROMPTS) + 2)
    long_pool   = LONG_PROMPTS  * (n_long   // len(LONG_PROMPTS)   + 2)

    rng.shuffle(short_pool)
    rng.shuffle(medium_pool)
    rng.shuffle(long_pool)

    prompts = (short_pool[:n_short] + medium_pool[:n_medium] + long_pool[:n_long])
    rng.shuffle(prompts)

    # Poisson inter-arrival times
    requests = []
    current_time = 0.0
    for i, prompt in enumerate(prompts):
        inter = rng.expovariate(arrival_rate_rps)
        current_time += inter
        req = RequestState(
            request_id=f"req_{i:04d}",
            prompt=prompt,
            prompt_len=len(prompt.split()),
            arrival_time=current_time,
        )
        requests.append(req)

    return requests


def save_workload(requests: List[RequestState], path: str):
    data = [
        {
            "request_id": r.request_id,
            "prompt": r.prompt,
            "prompt_len": r.prompt_len,
            "arrival_time": r.arrival_time,
        }
        for r in requests
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_workload(path: str) -> List[RequestState]:
    with open(path) as f:
        data = json.load(f)
    return [
        RequestState(
            request_id=d["request_id"],
            prompt=d["prompt"],
            prompt_len=d["prompt_len"],
            arrival_time=d["arrival_time"],
        )
        for d in data
    ]
