from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from swarm_probe.answers import extract_gold


@dataclass(frozen=True)
class Problem:
    qid: str
    question: str
    gold: str


def load_gsm8k(n_problems: int, seed: int) -> list[Problem]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    order = np.random.default_rng(seed).permutation(len(dataset))[:n_problems]
    return [
        Problem(
            qid=f"gsm8k-test-{int(index)}",
            question=dataset[int(index)]["question"],
            gold=extract_gold(dataset[int(index)]["answer"]),
        )
        for index in order
    ]
