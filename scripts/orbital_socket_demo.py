import argparse
import json
import os
import sys
import traceback

from swarmshield.orbital.socket_transport import project_path, run_experiment, save, summarize


def main():
    parser = argparse.ArgumentParser(description="CPU-only measured adapter socket transfers; no training")
    parser.add_argument("--config", default="configs/orbital_sft/socket_transport.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-local-cpu", action="store_true", help="Allow this CPU-only demo on a standalone workstation")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_local_cpu:
        parser.error("Run through a CPU Slurm allocation; use infra/rivanna/socket_transport.sbatch")
    output = project_path(args.output)
    try:
        if args.report_only:
            result = summarize(output)
        else:
            if output.exists():
                parser.error("Output already exists; choose a fresh run identity")
            config = json.loads(project_path(args.config).read_text())
            result = run_experiment(config, output)
    except Exception:
        if output.exists():
            save(output / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    print(json.dumps({"status": result["status"], "passed_checks": sum(result["checks"].values()),
                      "total_checks": len(result["checks"]), "report": str(output / "report.md")}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
