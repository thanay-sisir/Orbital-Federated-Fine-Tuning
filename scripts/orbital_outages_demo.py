import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from swarmshield.orbital.outages import OutageEngine, audit, plan_outage_round


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="docs/orbital_sft/evidence/outages_plot_data.json")
    parser.add_argument("--output", default="results/orbital/outage-cpu-demo")
    args = parser.parse_args()
    data = json.loads((ROOT / args.bundle).read_text())
    selected = data["selected"]
    settings = SimpleNamespace(rate_bps=selected["rate_bps"], chunk_bytes=1048576,
        framing_bytes=64, ack_bytes=32, acquisition_seconds=1, byte_budget=100000000000,
        queue_bytes=1073741824, min_participants=2)
    payload = selected["pending_queue_upper_bound_per_direction_bytes"] // 5
    outcomes = []
    for label, policy in (("no_outage_control", "fixed"), ("fixed", "fixed"), ("retry_backoff", "retry_backoff"), ("feasible_fair", "feasible_fair")):
        trace = data["trace"] if label != "no_outage_control" else {**data["trace"], "profile": "none", "events": []}
        engine = OutageEngine(data["geometry"]["contacts"], settings, trace, 5, 86400)
        age, accepted, performed = [0] * 5, 0, 0
        for round_id in range(6):
            engine, tasks, summary = plan_outage_round(engine, 5, round_id * 14400, (round_id + 1) * 14400,
                payload, selected["step_seconds"], 4, policy, age, round_id)
            accepted += summary["accepted_steps"]
            performed += summary["scheduled_steps"]
            members = set(summary["accepted_clients"])
            age = [0 if c in members else age[c] + 1 for c in range(5)]
        expected = next(r for r in data["rows"] if r["geometry"] == selected["geometry"] and r["rate_bps"] == selected["rate_bps"]
            and r["step_seconds"] == selected["step_seconds"] and r["policy"] == policy
            and r["profile"] == ("none" if label == "no_outage_control" else selected["profile"])
            and r["fault_seed"] == (101 if label == "no_outage_control" else selected["fault_seed"]))
        assert accepted == expected["accepted_steps"] and performed == expected["performed_steps"]
        assert engine.wire_bytes == expected["wire_bytes"] and engine.mission_clock == 86400
        outcomes.append({"condition": label, "planned_performed_steps": performed,
            "planned_accepted_steps": accepted, **audit(engine, 86400)})
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain inside project")
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "passed", "conditions": outcomes,
              "scope": "Standard-library CPU replay of archived nominal geometry and exogenous outages. No model training or quality inference."}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
