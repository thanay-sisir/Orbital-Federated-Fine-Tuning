import argparse
import contextlib
import json
import subprocess
import time

from swarmshield.orbital.checkpoint import append, atomic_json, read
from swarmshield.orbital.config import digest, load, project_path
from swarmshield.orbital.continuation import completed_adapter
from swarmshield.orbital.runner import make_run, require_allocation
from swarmshield.posttraining.benchmarks import prompt_ids
from swarmshield.posttraining.data import rows
from swarmshield.posttraining.evaluation import evaluate
from swarmshield.posttraining.model_loader import install, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--control", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    cfg.training.workers = 1
    require_allocation(cfg)
    selection_path = project_path("results/orbital/comparison-22406222/comparison.json")
    selection = read(selection_path)
    if selection["selected_by_validation"] != args.selected:
        raise ValueError("Selected run differs from frozen validation decision")
    suite = project_path("artifacts/benchmarks/ifeval-v1")
    protocol = read(suite / "manifest.json")
    if digest(suite / "prompts.jsonl") != protocol["prompts_sha256"]:
        raise ValueError("IFEval sample checksum mismatch")
    candidates = [("base", None, None)]
    for name in (args.selected, args.control):
        adapter, lineage = completed_adapter(name, cfg)
        candidates.append((name, adapter, lineage))
    path = make_run(cfg, args.run_id, "heldout_test_and_ifeval")
    atomic_json(path / "protocol.json", {
        **protocol, "selection_sha256": digest(selection_path),
        "selection_frozen_before_test": args.selected,
        "script_sha256": digest(project_path("scripts/orbital_final_evaluate.py")),
        "scoring_script_sha256": digest(project_path("scripts/orbital_ifeval_score.py")),
    })
    started = time.perf_counter()
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model = load_model(cfg, 0)
        tokenizer = AutoTokenizer.from_pretrained(project_path(cfg.model.path), local_files_only=True)
        prompts = rows(suite / "prompts.jsonl")
        data = project_path(cfg.data.prepared_path)
        results = {}
        for index, (name, adapter, lineage) in enumerate(candidates):
            if adapter is not None:
                install(model, load_file(str(adapter)))
            context = model.disable_adapter() if adapter is None else contextlib.nullcontext()
            with context:
                model.eval()
                item = {"lineage": lineage, "validation": evaluate(model, rows(data / "validation.jsonl"), 0), "test": evaluate(model, rows(data / "test.jsonl"), 0)}
                results[name] = item
                atomic_json(path / "results.partial.json", results)
                responses = path / f"responses-{index}.jsonl"
                capped = 0
                for count, row in enumerate(prompts, 1):
                    ids = prompt_ids(tokenizer, row["prompt"], 0)
                    with torch.no_grad():
                        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=protocol["max_new_tokens"], use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    tail = output[0, ids.shape[1]:]
                    capped += int(len(tail) == protocol["max_new_tokens"])
                    append(responses, {"key": row["key"], "prompt": row["prompt"], "response": tokenizer.decode(tail, skip_special_tokens=True), "generated_tokens": len(tail), "at_token_limit": len(tail) == protocol["max_new_tokens"]})
                    atomic_json(path / "progress.json", {"candidate": name, "ifeval_completed": count, "ifeval_total": len(prompts)})
                scored = path / f"ifeval-{index}.json"
                subprocess.run([str(project_path(".venv-orbital-score") / "bin/python"), str(project_path("scripts/orbital_ifeval_score.py")), "--responses", str(responses), "--output", str(scored)], check=True)
                item["ifeval"] = {**read(scored), "at_token_limit": capped, "sample_size": len(prompts)}
                atomic_json(path / "results.partial.json", results)
        atomic_json(path / "results.json", {"selected_by_prior_validation": args.selected, "results": results, "selection_uses_test": False})
        atomic_json(path / "health.json", {"status": "completed", "seconds": time.perf_counter() - started})
    except BaseException as error:
        atomic_json(path / "health.json", {"status": "failed", "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
