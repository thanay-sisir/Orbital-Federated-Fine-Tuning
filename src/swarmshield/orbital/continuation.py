from .checkpoint import read
from .config import Settings, digest, project_path


def completed_adapter(run_dir, cfg, previous_epoch=None):
    path = project_path(run_dir)
    manifest = read(path / "manifest.json")
    parent = Settings.model_validate(manifest["config"])
    if read(path / "health.json")["status"] != "completed":
        raise ValueError("Parent run has not completed")
    if parent.model != cfg.model or parent.data != cfg.data or parent.seed != cfg.seed:
        raise ValueError("Parent model, dataset or seed differs")
    coverage = read(path / "coverage.json")
    if not coverage["all_clients_aggregated"] or coverage["processed_examples"] != coverage["shard_examples"]:
        raise ValueError("Parent epoch coverage is incomplete")
    if previous_epoch is not None and parent.training.epoch_number != previous_epoch:
        raise ValueError("Parent epoch number differs")
    checkpoint = read(path / "checkpoint.json")
    adapter = project_path(checkpoint["global_adapter"])
    if not adapter.is_relative_to(path) or digest(adapter) != checkpoint["global_sha256"]:
        raise ValueError("Parent adapter checksum or location differs")
    return adapter, {
        "parent_run": str(path),
        "parent_epoch": parent.training.epoch_number,
        "adapter_sha256": digest(adapter),
        "parent_manifest_sha256": digest(path / "manifest.json"),
        "optimizer": "reset AdamW; retain aggregated adapter weights",
    }
