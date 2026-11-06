from .checkpoint import read
from .config import project_path


def report(path):
    path = project_path(path)
    manifest = read(path / "manifest.json")
    health = read(path / "health.json")
    evaluation = read(path / "evaluation.json") if (path / "evaluation.json").exists() else {}
    text = (
        "# Orbital federated SFT run\n\n"
        f"Status: {health['status']}. Virtual clients: {manifest['virtual_clients']}; "
        f"physical GPUs: {manifest['physical_gpus']}.\n\n"
        "Geometry, radio capacity and onboard compute time are synthetic assumptions. "
        "Actual model updates execute on allocated Earth-based GPUs.\n\n"
        f"Evaluation: `{evaluation}`. Answer loss is not task capability evidence.\n\n"
        "See metrics.csv, events.jsonl, transfers.jsonl, client_updates.jsonl, resources.json "
        "and manifest.json for the two clocks, byte counts and provenance. "
        "No algorithmic novelty, quality improvement or onboard feasibility is implied.\n"
    )
    (path / "report.md").write_text(text)
    return path / "report.md"
