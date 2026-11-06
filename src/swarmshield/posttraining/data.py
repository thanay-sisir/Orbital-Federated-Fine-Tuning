import hashlib
import json
import random
from collections import Counter

from swarmshield.orbital.checkpoint import append, atomic_json, read
from swarmshield.orbital.config import digest, project_path


def encode_conversation(tokenizer, messages, max_length, thinking=False):
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with assistant")
    expected = "user"
    for index, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Only nonempty text messages supported")
        if index == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError("Expected alternating user/assistant messages")
        expected = "assistant" if role == "user" else "user"
    kwargs = {"tokenize": False, "enable_thinking": thinking}
    text = tokenizer.apply_chat_template(messages, **kwargs)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:index], add_generation_prompt=True, **kwargs
        )
        completed = tokenizer.apply_chat_template(messages[: index + 1], **kwargs)
        if not text.startswith(prefix) or not text.startswith(completed):
            raise ValueError("Template is not prefix-stable; review assistant masks")
        spans.append((len(prefix), len(completed)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    labels = [
        token
        if any(start >= left and end <= right and end > start for left, right in spans)
        else -100
        for token, (start, end) in zip(ids, encoded["offset_mapping"])
    ]
    supervised = sum(token != -100 for token in labels[1:])
    if supervised == 0:
        raise ValueError("No supervised next-token targets")
    if len(ids) > max_length:
        return None, {"reason": "overlength", "tokens": len(ids), "supervised_tokens": supervised}
    return {"input_ids": ids, "labels": labels, "supervised_tokens": supervised}, None


def prepare(cfg):
    from transformers import AutoTokenizer

    cfg.ready(training=True)
    output = project_path(cfg.data.prepared_path)
    if output.exists():
        raise FileExistsError("Prepared dataset is immutable; choose a fresh destination")
    raw = project_path(cfg.data.raw_path)
    provenance = read(raw.with_suffix(".manifest.json"))
    if (provenance["identity"], provenance["revision"], provenance["sha256"]) != (
        cfg.data.identity,
        cfg.data.revision,
        digest(raw),
    ):
        raise ValueError("Raw dataset provenance mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        project_path(cfg.model.path), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Offset-based masking needs a fast tokenizer")
    output.mkdir(parents=True)
    seen, rows, counts = set(), [], Counter()
    try:
        with raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                source = row.get("source")
                if source not in cfg.data.source_terms:
                    counts["excluded_source"] += 1
                    continue
                messages = row.get("messages", [])
                canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                identity = hashlib.sha256(canonical.encode()).hexdigest()
                if identity in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(identity)
                try:
                    encoded, dropped = encode_conversation(
                        tokenizer, messages, cfg.data.max_length, cfg.data.thinking
                    )
                except (ValueError, KeyError, TypeError):
                    counts["malformed_or_mask_rejected"] += 1
                    continue
                if dropped:
                    counts["overlength"] += 1
                    counts["lost_supervised_tokens"] += dropped["supervised_tokens"]
                    continue
                split_number = (
                    int(hashlib.sha256(f"{cfg.seed}:{identity}".encode()).hexdigest()[:16], 16)
                    / 2**64
                )
                split = (
                    "test"
                    if split_number < cfg.data.test_fraction
                    else "validation"
                    if split_number < cfg.data.test_fraction + cfg.data.validation_fraction
                    else "train"
                )
                rows.append(
                    {
                        **encoded,
                        "id": row.get("id", identity),
                        "digest": identity,
                        "source": source,
                        "split": split,
                    }
                )
                counts["accepted"] += 1
                if len(rows) == cfg.data.max_examples:
                    break
        rng = random.Random(cfg.seed)
        train = [row for row in rows if row["split"] == "train"]
        rng.shuffle(train)
        if cfg.data.partition == "source_skew":
            train.sort(key=lambda row: row["source"])
            width = max(1, (len(train) + cfg.training.clients - 1) // cfg.training.clients)
            shards = [train[i * width : (i + 1) * width] for i in range(cfg.training.clients)]
        else:
            shards = [train[i :: cfg.training.clients] for i in range(cfg.training.clients)]
        if any(not shard for shard in shards):
            raise ValueError("Every client needs a nonempty shard")
        for index, shard in enumerate(shards):
            for row in shard:
                append(output / f"client-{index}.jsonl", row)
        for split in ("validation", "test"):
            selected = [row for row in rows if row["split"] == split]
            if not selected:
                raise ValueError(f"Empty held-out {split} split; increase sample")
            for row in selected:
                append(output / f"{split}.jsonl", row)
        mask_samples = [
            {
                "id": row["id"],
                "tokens": tokenizer.convert_ids_to_tokens(row["input_ids"]),
                "supervised": [v != -100 for v in row["labels"]],
            }
            for row in rows[:5]
        ]
        atomic_json(output / "mask_audit.json", mask_samples)
        atomic_json(
            output / "manifest.json",
            {
                "data": cfg.data.model_dump(),
                "model": cfg.model.model_dump(),
                "seed": cfg.seed,
                "clients": cfg.training.clients,
                "counts": dict(counts),
                "licence_review_sha256": digest(project_path(cfg.data.licence_review)),
                "shard_examples": [len(shard) for shard in shards],
                "sha256": {p.name: digest(p) for p in sorted(output.glob("*.jsonl"))},
                "overlap_audit": "Exact conversation dedup before hash splits; external benchmark overlap unknown",
            },
        )
    except BaseException:
        atomic_json(output / "FAILED.json", {"status": "incomplete; choose a new output path"})
        raise
    return output


def verify_prepared(cfg):
    path = project_path(cfg.data.prepared_path)
    if (path / "FAILED.json").exists():
        raise ValueError("Incomplete prepared data")
    manifest = read(path / "manifest.json")
    if (manifest["data"], manifest["model"], manifest["seed"], manifest["clients"]) != (
        cfg.data.model_dump(),
        cfg.model.model_dump(),
        cfg.seed,
        cfg.training.clients,
    ):
        raise ValueError("Prepared data contract mismatch")
    for name, checksum in manifest["sha256"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or digest(target) != checksum:
            raise ValueError("Prepared shard checksum mismatch")
    return manifest


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]
