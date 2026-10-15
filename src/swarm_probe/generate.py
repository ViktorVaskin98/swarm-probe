from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarm_probe.answers import extract_prediction, is_correct
from swarm_probe.data import Problem

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class SamplingConfig:
    n_samples: int
    greedy: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    batch_problems: int
    seed: int


@dataclass(frozen=True)
class Completion:
    qid: str
    model_id: str
    sample_idx: int
    greedy: bool
    text: str
    prediction: str | None
    correct: bool
    prompt_tokens: int
    generated_tokens: int
    hit_max_tokens: bool
    token_ids: list[int]


class Generator:
    def __init__(self, model_id: str, dtype: torch.dtype, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device).eval()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def run(self, problems: Sequence[Problem], config: SamplingConfig) -> Iterator[Completion]:
        for start in range(0, len(problems), config.batch_problems):
            batch = problems[start : start + config.batch_problems]
            torch.manual_seed(config.seed + start)
            yield from self._complete_batch(batch, config)

    @torch.inference_mode()
    def _complete_batch(self, batch: Sequence[Problem], config: SamplingConfig) -> list[Completion]:
        n_return = 1 if config.greedy else config.n_samples
        encoded = self.tokenizer(
            [self.build_prompt(p.question) for p in batch], return_tensors="pt", padding=True
        ).to(self.device)
        sampling = (
            {"do_sample": False}
            if config.greedy
            else {"do_sample": True, "temperature": config.temperature, "top_p": config.top_p}
        )
        output = self.model.generate(
            **encoded,
            **sampling,
            max_new_tokens=config.max_new_tokens,
            num_return_sequences=n_return,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = output[:, encoded["input_ids"].shape[1] :]
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        stop_ids = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id}
        completions: list[Completion] = []
        for row, tokens in enumerate(generated.tolist()):
            problem = batch[row // n_return]
            kept = [t for t in tokens if t not in stop_ids]
            text = self.tokenizer.decode(kept, skip_special_tokens=True)
            prediction = extract_prediction(text)
            completions.append(
                Completion(
                    qid=problem.qid,
                    model_id=self.model_id,
                    sample_idx=-1 if config.greedy else row % n_return,
                    greedy=config.greedy,
                    text=text,
                    prediction=prediction,
                    correct=is_correct(prediction, problem.gold),
                    prompt_tokens=prompt_lengths[row // n_return],
                    generated_tokens=len(kept),
                    hit_max_tokens=len(kept) >= config.max_new_tokens,
                    token_ids=kept,
                )
            )
        return completions
