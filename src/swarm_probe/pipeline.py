import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from swarm_probe.data import Problem, load_gsm8k
from swarm_probe.generate import Generator, SamplingConfig

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    dtype: str
    n_samples: int
    batch_problems: int


@dataclass(frozen=True)
class GenerationPlan:
    n_problems: int
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    models: list[ModelRun]

    @classmethod
    def from_json(cls, path: Path) -> "GenerationPlan":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["models"] = [ModelRun(**m) for m in raw["models"]]
        return cls(**raw)


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _run_one(
    generator: Generator, problems: list[Problem], config: SamplingConfig, out_path: Path
) -> dict[str, float]:
    started = time.time()
    n_correct = n_total = n_truncated = 0
    with out_path.open("w", encoding="utf-8") as sink:
        for completion in generator.run(problems, config):
            sink.write(json.dumps(asdict(completion), ensure_ascii=False) + "\n")
            n_total += 1
            n_correct += completion.correct
            n_truncated += completion.hit_max_tokens
    return {
        "accuracy": n_correct / n_total,
        "truncated_share": n_truncated / n_total,
        "n_completions": n_total,
        "seconds": round(time.time() - started, 1),
    }


def run_generation(plan: GenerationPlan, out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_gsm8k(plan.n_problems, plan.seed)
    (out_dir / "problems.jsonl").write_text(
        "".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in problems),
        encoding="utf-8",
    )
    summary: dict[str, dict] = {}
    for run in plan.models:
        generator = Generator(run.model_id, DTYPES[run.dtype])
        slug = _slug(run.model_id)
        summary[slug] = {"model_id": run.model_id, "n_params": generator.n_params}
        modes = [("greedy", True, 1)]
        if run.n_samples > 0:
            modes.append(("sampled", False, run.n_samples))
        for mode, greedy, n_samples in modes:
            config = SamplingConfig(
                n_samples=n_samples,
                greedy=greedy,
                temperature=plan.temperature,
                top_p=plan.top_p,
                max_new_tokens=plan.max_new_tokens,
                batch_problems=run.batch_problems * (max(run.n_samples, 1) if greedy else 1),
                seed=plan.seed,
            )
            stats = _run_one(generator, problems, config, out_dir / f"{slug}__{mode}.jsonl")
            summary[slug][mode] = stats
            print(f"[{slug}] {mode}: {stats}", flush=True)
        del generator
        torch.cuda.empty_cache()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
