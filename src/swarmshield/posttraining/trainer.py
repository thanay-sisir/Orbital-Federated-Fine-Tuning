import random
import time

from swarmshield.orbital.checkpoint import atomic_json
from swarmshield.orbital.config import Settings, digest, project_path

from .data import rows
from .evaluation import evaluate
from .model_loader import adapter_state, fixed_a_hash, frozen_hash, install, load_model

_MODEL = None
_CONFIG = None


def local_steps(cfg, task):
    steps = task.get("local_steps", cfg.training.local_steps)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= cfg.training.local_steps:
        raise ValueError("Task local steps exceed the configured budget")
    return steps


def validate_gradients(named_parameters):
    import torch

    for name, parameter in named_parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("Unexpected or nonfinite gradient")
        if parameter.grad is not None and not parameter.requires_grad:
            raise ValueError(f"Frozen parameter received a gradient: {name}")


def execute(task):
    import torch
    from safetensors.torch import load_file, save_file

    global _MODEL, _CONFIG
    cfg = Settings.model_validate(task["config"])
    device = task["device"]
    if _MODEL is None or _CONFIG != (cfg.fingerprint(), device):
        _MODEL = load_model(cfg, device)
        _CONFIG = (cfg.fingerprint(), device)
    model = _MODEL
    output = project_path(task["output"])
    output.mkdir(parents=True, exist_ok=False)
    initial = adapter_state(model) if task.get("initial") else load_file(task["global_adapter"])
    install(model, initial)
    client, cursor = task["client"], task["cursor"]
    seed = cfg.training_seed + client * 100003 + task["round"] + (cfg.training.epoch_number - 1) * 1009
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    examples = rows(project_path(cfg.data.prepared_path) / f"client-{client}.jsonl")
    if cfg.training.full_shard_pass and cfg.training.epoch_number > 1:
        random.Random(seed).shuffle(examples)
    validation = rows(project_path(cfg.data.prepared_path) / "validation.jsonl")[
        : cfg.training.validation_examples
    ]
    pilot = task.get("pilot", False)
    base_hash = frozen_hash(model) if pilot else None
    before_eval = evaluate(model, validation, device)
    if task.get("initial"):
        with model.disable_adapter():
            atomic_json(
                output / "untouched_base_validation.json", evaluate(model, validation, device)
            )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    tokens, unique, losses = 0, set(), []
    sample_count = local_steps(cfg, task) * cfg.training.accumulation
    if cfg.training.full_shard_pass:
        if cursor != 0 or sample_count < len(examples):
            raise ValueError("Full shard pass requires a fresh cursor and sufficient local steps")
        sample_count = len(examples)
    step_count = (sample_count + cfg.training.accumulation - 1) // cfg.training.accumulation
    for step in range(step_count):
        optimizer.zero_grad(set_to_none=True)
        batch_count = min(cfg.training.accumulation, sample_count - step * cfg.training.accumulation)
        for _ in range(batch_count):
            row = examples[cursor % len(examples)]
            cursor += 1
            unique.add(row["digest"])
            ids = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / batch_count).backward()
            tokens += row["supervised_tokens"]
            losses.append(float(loss.detach()))
        validate_gradients(model.named_parameters())
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.training.grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    state = adapter_state(model)
    changed = any(not torch.equal(state[name], initial[name]) for name in state if "lora_B" in name)
    if not changed:
        raise ValueError("Training did not change B")
    if any(not torch.equal(state[name], initial[name]) for name in state if "lora_A" in name):
        raise ValueError("Fixed A changed")
    save_file(state, str(output / "adapter.safetensors"))
    save_file({k: v for k, v in state.items() if "lora_B" in k}, str(output / "upload.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "python_rng": random.getstate(),
            "cursor": cursor,
        },
        output / "local_state.pt",
    )
    model.save_pretrained(output / "peft", safe_serialization=True)
    after_eval = evaluate(model, validation, device)
    if pilot:
        if frozen_hash(model) != base_hash:
            raise ValueError("Frozen base changed")
        del optimizer
        _MODEL = None
        del model
        torch.cuda.empty_cache()
        _MODEL = load_model(cfg, device)
        model = _MODEL
    install(model, load_file(str(output / "adapter.safetensors")))
    reloaded = evaluate(model, validation, device)
    if abs(reloaded["answer_loss"] - after_eval["answer_loss"]) > cfg.training.reload_tolerance:
        raise ValueError("Adapter reload evaluation mismatch")
    if pilot and frozen_hash(model) != base_hash:
        raise ValueError("Frozen base changed")
    result = {
        "client": client,
        "version": task["version"],
        "cursor": cursor,
        "base_identity": cfg.model.identity + "@" + cfg.model.revision,
        "unique_examples": len(unique),
        "example_digests": sorted(unique),
        "supervised_tokens": tokens,
        "steps": step_count,
        "loss": sum(losses) / len(losses),
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "device_name": torch.cuda.get_device_name(device),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "validation_before": before_eval,
        "validation_after": after_eval,
        "reload_validation": reloaded,
        "adapter_changed": changed,
        "base_frozen_sha256": base_hash,
        "base_unchanged": True if pilot else "pilot gate only",
        "adapter_path": str(output / "adapter.safetensors"),
        "upload_path": str(output / "upload.safetensors"),
        "upload_sha256": digest(output / "upload.safetensors"),
        "output": str(output),
        "fixed_A": True,
        "fixed_A_sha256": fixed_a_hash(state),
        "adapter_contract": cfg.model.model_dump(),
    }
    atomic_json(output / "result.json", result)
    return result


def initialize(config, device, output, parent_adapter=None):
    from safetensors.torch import save_file

    global _MODEL, _CONFIG
    cfg = Settings.model_validate(config)
    if _MODEL is None or _CONFIG != (cfg.fingerprint(), device):
        _MODEL = load_model(cfg, device)
        _CONFIG = (cfg.fingerprint(), device)
    path = project_path(output)
    path.mkdir(parents=True, exist_ok=False)
    if parent_adapter is not None:
        from safetensors.torch import load_file

        install(_MODEL, load_file(str(project_path(parent_adapter))))
    state = adapter_state(_MODEL)
    save_file(state, str(path / "adapter.safetensors"))
    save_file({k: v for k, v in state.items() if "lora_B" in k}, str(path / "B.safetensors"))
    _MODEL.save_pretrained(path / "peft", safe_serialization=True)
    return str(path / "adapter.safetensors")


def assess(config, device, adapter, output, final=False, full_validation=False):
    from safetensors.torch import load_file

    cfg = Settings.model_validate(config)
    if _MODEL is None:
        raise RuntimeError("Initialize worker before evaluation")
    install(_MODEL, load_file(adapter))
    validation = rows(project_path(cfg.data.prepared_path) / "validation.jsonl")
    if not full_validation:
        validation = validation[: cfg.training.validation_examples]
    result = {"validation": evaluate(_MODEL, validation, device)}
    if final:
        if cfg.training.evaluate_test:
            result["test"] = evaluate(
                _MODEL, rows(project_path(cfg.data.prepared_path) / "test.jsonl"), device
            )
        path = project_path(output)
        path.mkdir(parents=True, exist_ok=False)
        _MODEL.save_pretrained(path, safe_serialization=True)
        atomic_json(
            path / "inference_contract.json",
            {
                "base": cfg.model.model_dump(),
                "factor_policy": "fixed_A_train_B",
                "evaluation": result,
                "tokenizer": cfg.model.path,
                "usage": "Load pinned base, then PeftModel.from_pretrained(base, this directory)",
            },
        )
    return result
