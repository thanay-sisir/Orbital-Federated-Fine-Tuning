import argparse
import contextlib
import time

from swarmshield.orbital.checkpoint import append, atomic_json, read
from swarmshield.orbital.config import digest, load, project_path
from swarmshield.orbital.continuation import completed_adapter
from swarmshield.orbital.runner import make_run, require_allocation
from swarmshield.posttraining.benchmarks import arc_prediction, gsm_prediction
from swarmshield.posttraining.data import rows
from swarmshield.posttraining.evaluation import evaluate
from swarmshield.posttraining.model_loader import install, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--benchmark-dir", default="artifacts/benchmarks/development-v1")
    args = parser.parse_args()
    cfg = load(args.config)
    cfg.training.workers = 1
    require_allocation(cfg)
    bench = project_path(args.benchmark_dir)
    protocol = read(bench / "manifest.json")
    datasets = {}
    for source in protocol["sources"]:
        target = (bench / source["file"]).resolve()
        if not target.is_relative_to(bench) or digest(target) != source["sha256"]:
            raise ValueError("Benchmark checksum differs")
        datasets[source["task"]] = rows(target)
    candidates = [(name, *completed_adapter(name, cfg)) for name in args.candidate]
    path = make_run(cfg, args.run_id, "base_adapter_comparison")
    atomic_json(path / "protocol.json", {
        **protocol, "evaluator_sha256": digest(project_path("scripts/orbital_compare.py")),
        "benchmark_manifest_sha256": digest(bench / "manifest.json"),
    })
    started = time.perf_counter()
    try:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model = load_model(cfg, 0)
        tokenizer = AutoTokenizer.from_pretrained(project_path(cfg.model.path), local_files_only=True)
        validation = rows(project_path(cfg.data.prepared_path) / "validation.jsonl")
        results = {}
        for name, adapter, lineage in [("base", None, None), *candidates]:
            if adapter is not None:
                install(model, load_file(str(adapter)))
            context = model.disable_adapter() if adapter is None else contextlib.nullcontext()
            with context:
                model.eval()
                metrics = {"validation": evaluate(model, validation, 0), "lineage": lineage}
                if args.final_test:
                    metrics["test"] = evaluate(model, rows(project_path(cfg.data.prepared_path) / "test.jsonl"), 0)
                atomic_json(path / "progress.json", {"candidate": name, "stage": "benchmarks", "validation": metrics["validation"]})
                results[name] = metrics
                atomic_json(path / "comparison.partial.json", results)
                for task, examples in datasets.items():
                    correct, truncated = 0, 0
                    for row in examples:
                        result = (
                            arc_prediction(model, tokenizer, row, 0) if task == "arc"
                            else gsm_prediction(model, tokenizer, row, 0, protocol["max_new_tokens"])
                        )
                        correct += result["correct"]
                        truncated += result.get("at_token_limit", False)
                        append(path / "predictions.jsonl", {"candidate": name, "task": task, "index": row["index"], **result})
                    n, z = len(examples), 1.96
                    accuracy = correct / n
                    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
                    half = z * ((accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
                    metrics[task] = {"accuracy": accuracy, "correct": correct, "examples": n, "at_token_limit": truncated, "wilson_95": [center - half, center + half]}
                    atomic_json(path / "comparison.partial.json", results)
            atomic_json(path / "comparison.partial.json", results)
        winner = min(results, key=lambda name: results[name]["validation"]["answer_loss"])
        atomic_json(path / "comparison.json", {
            "results": results, "selected_by_validation": winner,
            "selection_metric": "token-weighted answer loss over all 344 validation conversations",
            "test_split_evaluated": args.final_test,
            "benchmark_caution": "Custom development protocol; contamination unknown; not standard leaderboard scores",
        })
        atomic_json(path / "health.json", {"status": "completed", "seconds": time.perf_counter() - started})
    except BaseException as error:
        atomic_json(path / "health.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
