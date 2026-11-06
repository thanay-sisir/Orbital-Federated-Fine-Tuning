import argparse

import numpy as np
from safetensors.numpy import load_file

from swarmshield.orbital.checkpoint import atomic_json
from swarmshield.orbital.config import digest, project_path
from swarmshield.orbital.quantized_update import decode, encode


def norm(state):
    return sum(float(np.square(v.astype(np.float64)).sum()) for v in state.values()) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    runs = ("reference-22367142", "epoch-22406221")
    residual, transmitted, exact, ordinary = {}, {}, {}, {}
    results = []
    for index, name in enumerate(runs, 1):
        root = project_path("results/orbital") / name
        upload = root / "round-0000/client-0/upload.safetensors"
        parent_path = root / "initial/adapter.safetensors"
        current, parent = load_file(str(upload)), load_file(str(parent_path))
        delta = {key: value - parent[key] for key, value in current.items()}
        payload, residual = encode(delta, residual)
        reconstructed = decode(payload)
        baseline, _ = encode(delta)
        no_feedback = decode(baseline)
        for key in delta:
            exact[key] = exact.get(key, np.zeros_like(delta[key])) + delta[key]
            transmitted[key] = transmitted.get(key, np.zeros_like(delta[key])) + reconstructed[key]
            ordinary[key] = ordinary.get(key, np.zeros_like(delta[key])) + no_feedback[key]
        (output / f"update-{index}.orbq8").write_bytes(payload)
        results.append({
            "epoch": index, "upload_sha256": digest(upload), "parent_sha256": digest(parent_path),
            "float32_serialized_bytes": upload.stat().st_size, "int8_serialized_bytes": len(payload),
            "size_ratio": upload.stat().st_size / len(payload),
            "cumulative_relative_error_with_feedback": norm({k: exact[k] - transmitted[k] for k in exact}) / max(norm(exact), 1e-30),
            "cumulative_relative_error_without_feedback": norm({k: exact[k] - ordinary[k] for k in exact}) / max(norm(exact), 1e-30),
        })
    atomic_json(output / "result.json", {
        "status": "completed", "results": results,
        "scope": "CPU codec study on real saved B deltas; residual stays local and is not transmitted",
        "quality": "Not evaluated; historical updates were not trained using quantized aggregation",
        "code_sha256": digest(project_path("src/swarmshield/orbital/quantized_update.py")),
    })


if __name__ == "__main__":
    main()
