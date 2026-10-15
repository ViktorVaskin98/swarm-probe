import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from swarm_probe.generate import Generator
from swarm_probe.pipeline import DTYPES


@dataclass(frozen=True)
class FeaturePlan:
    model_id: str
    dtype: str
    completions_file: str
    prefixes: list[int]
    layers: list[int]
    batch_size: int

    @classmethod
    def from_json(cls, path: Path) -> "FeaturePlan":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Sample:
    qid: str
    sample_idx: int
    correct: bool
    prompt_ids: list[int]
    answer_ids: list[int]


class HiddenStateExtractor:
    def __init__(self, generator: Generator, plan: FeaturePlan) -> None:
        self.generator = generator
        self.plan = plan

    def prompt_ids(self, question: str) -> list[int]:
        prompt = self.generator.build_prompt(question)
        return self.generator.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()
    def extract(self, samples: Sequence[Sample]) -> dict[str, np.ndarray]:
        n_layers = len(self.plan.layers)
        hidden = self.generator.model.config.hidden_size
        states = {
            m: np.zeros((len(samples), n_layers, hidden), dtype=np.float16)
            for m in self.plan.prefixes
        }
        available = {m: np.zeros(len(samples), dtype=bool) for m in self.plan.prefixes}
        prefix_logprob = {m: np.full(len(samples), np.nan) for m in self.plan.prefixes}
        full_logprob = np.full(len(samples), np.nan)
        pad_id = self.generator.tokenizer.pad_token_id
        for start in range(0, len(samples), self.plan.batch_size):
            batch = samples[start : start + self.plan.batch_size]
            sequences = [s.prompt_ids + s.answer_ids for s in batch]
            width = max(len(seq) for seq in sequences)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            for row, seq in enumerate(sequences):
                input_ids[row, : len(seq)] = torch.tensor(seq)
                attention[row, : len(seq)] = 1
            output = self.generator.model(
                input_ids=input_ids.to(self.generator.device),
                attention_mask=attention.to(self.generator.device),
                output_hidden_states=True,
            )
            layer_stack = torch.stack([output.hidden_states[i] for i in self.plan.layers], dim=1)
            for row, sample in enumerate(batch):
                index = start + row
                n_prompt = len(sample.prompt_ids)
                n_answer = len(sample.answer_ids)
                answer = torch.tensor(sample.answer_ids, device=output.logits.device)
                answer_logits = output.logits[row, n_prompt - 1 : n_prompt - 1 + n_answer]
                token_lp = (
                    torch.log_softmax(answer_logits.float(), dim=-1)
                    .gather(1, answer.unsqueeze(1))
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
                if n_answer > 0:
                    full_logprob[index] = token_lp.mean()
                for m in self.plan.prefixes:
                    seen = n_answer if m < 0 else m
                    if seen > n_answer or (m < 0 and n_answer == 0):
                        continue
                    available[m][index] = True
                    position = n_prompt - 1 + seen
                    states[m][index] = layer_stack[row, :, position].float().cpu().numpy()
                    if seen > 0:
                        prefix_logprob[m][index] = token_lp[:seen].mean()
            print(f"features {start + len(batch)}/{len(samples)}", flush=True)
        arrays: dict[str, np.ndarray] = {
            "correct": np.array([s.correct for s in samples]),
            "qid": np.array([s.qid for s in samples]),
            "sample_idx": np.array([s.sample_idx for s in samples]),
            "answer_len": np.array([len(s.answer_ids) for s in samples]),
            "full_logprob": full_logprob,
            "layers": np.array(self.plan.layers),
        }
        for m in self.plan.prefixes:
            arrays[f"h_{m}"] = states[m]
            arrays[f"avail_{m}"] = available[m]
            arrays[f"prefix_logprob_{m}"] = prefix_logprob[m]
        return arrays


def run_features(plan: FeaturePlan, run_dir: Path, out_path: Path) -> None:
    problems = {}
    with (run_dir / "problems.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            problems[row["qid"]] = row["question"]
    generator = Generator(plan.model_id, DTYPES[plan.dtype])
    extractor = HiddenStateExtractor(generator, plan)
    prompt_cache: dict[str, list[int]] = {}
    samples: list[Sample] = []
    mismatched = 0
    with (run_dir / plan.completions_file).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["qid"] not in prompt_cache:
                prompt_cache[row["qid"]] = extractor.prompt_ids(problems[row["qid"]])
            prompt = prompt_cache[row["qid"]]
            mismatched += len(prompt) != row["prompt_tokens"]
            samples.append(
                Sample(
                    qid=row["qid"],
                    sample_idx=row["sample_idx"],
                    correct=row["correct"],
                    prompt_ids=prompt,
                    answer_ids=row["token_ids"],
                )
            )
    print(f"samples {len(samples)}, prompt length mismatches {mismatched}", flush=True)
    arrays = extractor.extract(samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
