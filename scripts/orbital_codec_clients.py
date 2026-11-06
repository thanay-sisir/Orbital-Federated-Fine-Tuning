import argparse

import numpy as np
from safetensors.numpy import load_file

from swarmshield.orbital.checkpoint import atomic_json
from swarmshield.orbital.config import digest, project_path
from swarmshield.orbital.quantized_update import encode, decode


def norm(values):
    return sum(float(np.square(v.astype(np.float64)).sum()) for v in values.values()) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for client in range(5):
        residual, exact, reconstructed, without = {}, {}, {}, {}
        for epoch, run in enumerate(("reference-22367142", "epoch-22406221"), 1):
            parent = project_path("results/orbital") / run
            path = parent / f"round-0000/client-{client}/upload.safetensors"
            old = load_file(str(parent / "initial/adapter.safetensors"))
            current = load_file(str(path))
            delta = {key: value - old[key] for key, value in current.items()}
            payload, residual = encode(delta, residual)
            recovered = decode(payload)
            plain, _ = encode(delta)
            recovered_plain = decode(plain)
            for key in delta:
                exact[key] = exact.get(key, np.zeros_like(delta[key])) + delta[key]
                reconstructed[key] = reconstructed.get(key, np.zeros_like(delta[key])) + recovered[key]
                without[key] = without.get(key, np.zeros_like(delta[key])) + recovered_plain[key]
            (output / f"client-{client}-epoch-{epoch}.orbq8").write_bytes(payload)
            results.append({"client": client, "epoch": epoch, "input_sha256": digest(path), "raw_serialized_bytes": path.stat().st_size, "int8_serialized_bytes": len(payload), "size_ratio": path.stat().st_size / len(payload), "cumulative_error_feedback": norm({k: exact[k] - reconstructed[k] for k in exact}) / max(norm(exact), 1e-30), "cumulative_error_no_feedback": norm({k: exact[k] - without[k] for k in exact}) / max(norm(exact), 1e-30)})
    atomic_json(output / "results.json", {"status": "completed", "results": results, "quality": "CPU codec tests across all five archived clients; no quantized learning-quality claim", "implementation_sha256": digest(project_path("src/swarmshield/orbital/quantized_update.py"))})


if __name__ == "__main__":
    main()
