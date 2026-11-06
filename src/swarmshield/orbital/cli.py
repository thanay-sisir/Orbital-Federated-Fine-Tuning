import argparse
import json
import sys

from .config import load, project_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="orbital-sft")
    sub = parser.add_subparsers(dest="operation", required=True)
    for command in ("validate", "prepare-data", "contacts", "pilot", "run", "mission-run"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        if command == "validate":
            item.add_argument("--ready", action="store_true")
        if command in ("pilot", "run", "mission-run"):
            item.add_argument("--run-id", required=True)
        if command == "run":
            item.add_argument("--parent-run")
            item.add_argument(
                "--resume", help="Explicit completed-round checkpoint; new run ID required"
            )
        if command == "mission-run":
            item.add_argument("--recoverable", action="store_true")
            item.add_argument("--resume", action="store_true", help="Resume the same recoverable run identity")
            item.add_argument("--fault-plan", help="Explicit process-interruption demonstration plan")
            item.add_argument("--outage-trace", help="Realized exogenous outages; available only to the execution backend")
    item = sub.add_parser("report")
    item.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "report":
            from .reporting import report

            result = report(project_path(args.run_dir))
        else:
            cfg = load(args.config)
            if args.operation == "run" and args.parent_run:
                from .config import Settings

                values = cfg.model_dump()
                values["training"]["parent_run"] = args.parent_run
                cfg = Settings.model_validate(values)
            if args.operation == "validate":
                if args.ready:
                    cfg.ready(training=True)
                result = {
                    "schema": "valid",
                    "execution_inputs": "checked" if args.ready else "not_checked",
                    "config_hash": cfg.fingerprint(),
                }
            elif args.operation == "contacts":
                from .contacts import generate

                result = generate(cfg)
            elif args.operation == "prepare-data":
                from swarmshield.posttraining.data import prepare

                result = prepare(cfg)
            elif args.operation == "mission-run":
                from .mission_runner import run_mission

                if args.recoverable:
                    from .resilient_mission import run_resilient_mission

                    result = run_resilient_mission(cfg, args.run_id, args.resume, args.fault_plan, outage_trace=args.outage_trace)
                else:
                    if args.resume or args.fault_plan or args.outage_trace:
                        raise ValueError("Mission resume and fault injection require --recoverable")
                    result = run_mission(cfg, args.run_id)
            else:
                from .runner import pilot, run

                result = (
                    pilot(cfg, args.run_id)
                    if args.operation == "pilot"
                    else run(cfg, args.run_id, args.resume)
                )
        print(json.dumps(result if isinstance(result, dict) else {"path": str(result)}, indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"orbital-sft: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
