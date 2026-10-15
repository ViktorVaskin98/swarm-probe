import sys
from pathlib import Path

MODE = "generate"
CONFIG_NAME = "smoke"
SOURCE_RUN = "full"


def locate_code_root() -> Path:
    for marker in Path("/kaggle/input").rglob("swarm_probe/__init__.py"):
        return marker.parent.parent.parent
    raise FileNotFoundError("swarm-probe code dataset is not attached")


def locate_run_dir(name: str) -> Path:
    for marker in Path("/kaggle/input").rglob(f"{name}/summary.json"):
        return marker.parent
    raise FileNotFoundError(f"generation output {name!r} is not attached")


def main() -> None:
    root = locate_code_root()
    sys.path.insert(0, str(root / "src"))

    import torch

    print("code root:", root, flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    for index in range(torch.cuda.device_count()):
        print("gpu", index, torch.cuda.get_device_name(index), flush=True)
    config_path = root / "configs" / f"{CONFIG_NAME}.json"
    working = Path("/kaggle/working")
    if MODE == "generate":
        from swarm_probe.pipeline import GenerationPlan, run_generation

        summary = run_generation(GenerationPlan.from_json(config_path), working / CONFIG_NAME)
        print("REQUIRED_OUTPUT summary", summary, flush=True)
    elif MODE == "features":
        from swarm_probe.features import FeaturePlan, run_features

        run_dir = locate_run_dir(SOURCE_RUN)
        out_path = working / f"{CONFIG_NAME}.npz"
        run_features(FeaturePlan.from_json(config_path), run_dir, out_path)
        print("REQUIRED_OUTPUT features", out_path, out_path.stat().st_size, flush=True)
    else:
        raise ValueError(f"unknown mode {MODE!r}")


if __name__ == "__main__":
    main()
