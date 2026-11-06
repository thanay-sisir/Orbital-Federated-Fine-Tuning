import json
import os
import random

from swarmshield.orbital.checkpoint import atomic_json
from swarmshield.orbital.config import digest, project_path


def main():
    os.environ["HF_HOME"] = str(project_path(".cache/huggingface"))
    os.environ["HF_DATASETS_CACHE"] = str(project_path(".cache/huggingface/datasets"))
    os.environ["TMPDIR"] = str(project_path(".cache/tmp"))
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = project_path("artifacts/benchmarks/development-v1")
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for name, repo, subset, split, limit in (
        ("arc", "allenai/ai2_arc", "ARC-Challenge", "validation", None),
        ("gsm8k", "openai/gsm8k", "main", "test", 100),
    ):
        revision = HfApi().dataset_info(repo).sha
        dataset = load_dataset(repo, subset, split=split, revision=revision)
        indices = list(range(len(dataset)))
        if limit:
            indices = sorted(random.Random(42).sample(indices, min(limit, len(indices))))
        path = output / f"{name}.jsonl"
        with path.open("x") as stream:
            for index in indices:
                stream.write(json.dumps({"index": index, **dataset[index]}, ensure_ascii=False) + "\n")
        sources.append({
            "task": name, "repo": repo, "revision": revision, "subset": subset,
            "split": split, "count": len(indices), "file": path.name, "sha256": digest(path),
        })
    atomic_json(output / "manifest.json", {
        "protocol": "development-v1; custom prompts/scoring; not leaderboard-comparable",
        "sources": sources, "seed": 42, "max_new_tokens": 256,
        "thinking": False, "selection_metric": "full Tulu validation answer loss only",
        "contamination": "External benchmark overlap with original pretraining is unknown",
    })
    print(output)


if __name__ == "__main__":
    main()
