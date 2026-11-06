import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from swarmshield.orbital.network import LinkEngine
from swarmshield.orbital.reliable_transfer import DurableReceiver, TransferSpec
from swarmshield.orbital.round_planner import plan_round


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="docs/orbital_sft/evidence/plot_data.json")
    parser.add_argument("--output", default="results/orbital/cpu-demo")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside project")
    output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((ROOT / args.bundle).read_text())
    settings = SimpleNamespace(rate_bps=1000000, chunk_bytes=1048576, framing_bytes=64,
                               ack_bytes=32, acquisition_seconds=1, queue_bytes=1073741824,
                               byte_budget=100000000000, min_participants=2)
    comparisons = []
    for policy in ("fixed", "feasible_fair"):
        engine = LinkEngine(evidence["contacts"]["contacts"], settings)
        age = [0] * 5
        accepted_steps = performed_steps = accepted_updates = 0
        for round_id in range(6):
            engine, tasks, summary = plan_round(engine, 5, round_id * 14400, (round_id + 1) * 14400,
                                                 6820712, 60, 4, policy, age, round_id)
            accepted = set(summary["accepted_clients"])
            accepted_steps += summary["accepted_steps"]
            performed_steps += sum(t["steps"] for t in tasks)
            accepted_updates += len(accepted)
            age = [0 if c in accepted else age[c] + 1 for c in range(5)]
        expected = next(r for r in evidence["runs"] if r["seed"] == 42 and r["policy"] == policy)
        for key, value in (("accepted_steps", accepted_steps), ("performed_steps", performed_steps), ("accepted_updates", accepted_updates), ("wire_bytes", engine.wire_bytes)):
            if expected[key] != value:
                raise AssertionError(f"CPU planner no longer reproduces measured {policy}/{key}")
        comparisons.append({"policy": policy, "accepted_steps": accepted_steps, "performed_steps": performed_steps,
                            "accepted_updates": accepted_updates, "wire_bytes": engine.wire_bytes})
    with tempfile.TemporaryDirectory(prefix="receiver-", dir=output) as work:
        payload = bytes(range(256)) * 1024
        spec = TransferSpec.from_bytes("demo", 0, 0, "synthetic-base", payload, 65536)
        receiver = DurableReceiver(Path(work) / "receiver.sqlite", "synthetic-base")
        receiver.begin(spec)
        receiver.put("demo", 0, payload[:65536])
        receiver.close()
        receiver = DurableReceiver(Path(work) / "receiver.sqlite", "synthetic-base")
        restored = receiver.missing("demo") == [1, 2, 3]
        for index in receiver.missing("demo"):
            receiver.put("demo", index, payload[index * 65536:(index + 1) * 65536])
        first = receiver.finalize("demo")
        second = receiver.finalize("demo")
        assert restored and first["newly_eligible"] and not second["newly_eligible"]
        receiver.close()
    result = {"status": "passed", "planning_reproduces_measured_delivery_counts": comparisons,
              "receiver_reopen_and_idempotency": True,
              "scope": "CPU scheduling replay and synthetic-byte receiver demonstration. No model training, quality evaluation or real radio communication."}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
