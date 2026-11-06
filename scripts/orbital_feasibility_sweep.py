import argparse
import itertools
import time

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, load, project_path
from swarmshield.orbital.feasibility import POLICIES, simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = project_path(args.output) / args.policy
    output.mkdir(parents=True, exist_ok=False)
    source = project_path("results/orbital/reference-22367142")
    cfg = load("configs/orbital_sft/five_gpu_one_epoch.yaml")
    contacts = read(source / "contacts.json")["contacts"]
    payload = source / "round-0000/global_B.safetensors"
    atomic_json(output / "manifest.json", {
        "policy": args.policy, "payload_bytes": payload.stat().st_size,
        "payload_sha256": digest(payload), "contacts_sha256": digest(source / "contacts.json"),
        "code_sha256": digest(project_path("src/swarmshield/orbital/feasibility.py")),
        "mission_seconds": 86400, "max_local_steps": 4, "minimum_participants": 2,
        "profiles": "Uncalibrated sensitivity assumptions; no onboard hardware claim",
        "quality": "not evaluated; CPU-only feasibility planning",
    })
    summaries = []
    started = time.perf_counter()
    for case_id, (rate, seconds, interval, station_count) in enumerate(itertools.product(
        (64000, 256000, 1000000, 10000000), (15, 60, 180), (1800, 3600, 7200, 14400), (1, 2)
    )):
        settings = cfg.network.model_copy(update={"rate_bps": rate, "min_participants": 2})
        names = {s.name for s in cfg.orbit.stations[:station_count]}
        selected = [c for c in contacts if c["station"] in names]
        result = simulate(selected, settings, 5, 86400, interval, payload.stat().st_size, seconds, policy=args.policy)
        atomic_json(output / f"case-{case_id:03d}.json", result)
        summaries.append({"case": case_id, "rate_bps": rate, "step_seconds": seconds, "round_seconds": interval, "stations": station_count, **{k: v for k, v in result.items() if k not in {"rounds", "receipts"}}})
        atomic_json(output / "progress.json", {"completed": len(summaries), "total": 96})
    atomic_json(output / "summary.json", summaries)
    atomic_json(output / "health.json", {"status": "completed", "cases": len(summaries), "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
