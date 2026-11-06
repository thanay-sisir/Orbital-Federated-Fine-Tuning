import argparse
import json
import os

from swarmshield.orbital.checkpoint import atomic_json
from swarmshield.orbital.config import digest, load, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["model", "data"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    cfg.ready()
    os.environ["HF_HOME"] = str(project_path(".cache/huggingface"))
    if args.operation == "model":
        from huggingface_hub import snapshot_download

        if cfg.model.tokenizer_revision != cfg.model.revision:
            raise ValueError("Staging helper requires colocated tokenizer and model revisions")
        target = project_path(cfg.model.path)
        if target.exists():
            raise FileExistsError("Model destination already exists")
        snapshot_download(
            cfg.model.identity,
            revision=cfg.model.revision,
            local_dir=target,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.jinja",
                "*.txt",
                "*.model",
                "LICENSE*",
                "README.md",
            ],
        )
        atomic_json(
            target / "snapshot_manifest.json",
            {
                "identity": cfg.model.identity,
                "revision": cfg.model.revision,
                "tokenizer_revision": cfg.model.tokenizer_revision,
                "sha256": {
                    str(p.relative_to(target)): digest(p)
                    for p in sorted(target.rglob("*"))
                    if p.is_file() and ".cache" not in p.parts
                },
            },
        )
    else:
        from datasets import load_dataset

        target = project_path(cfg.data.raw_path)
        if target.exists():
            raise FileExistsError("Raw dataset destination already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(
            cfg.data.identity, revision=cfg.data.revision, split="train", streaming=True
        )
        count = 0
        with target.open("x") as stream:
            for row in dataset:
                if row.get("source") in cfg.data.source_terms:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
        atomic_json(
            target.with_suffix(".manifest.json"),
            {
                "identity": cfg.data.identity,
                "revision": cfg.data.revision,
                "sha256": digest(target),
                "examples": count,
                "source_terms": cfg.data.source_terms,
            },
        )
    print(target)


if __name__ == "__main__":
    main()
