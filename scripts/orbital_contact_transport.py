import argparse
import json
import os
import sys
import traceback

from swarmshield.orbital.contact_transport import plan_study, run_case
from swarmshield.orbital.socket_transport import project_path, save


def main():
    parser = argparse.ArgumentParser(description="CPU-only saved-adapter socket replay under orbital contacts")
    parser.add_argument("--config", default="configs/orbital_sft/contact_transport.json")
    parser.add_argument("--case", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a CPU Slurm allocation")
    output = project_path(args.output)
    if output.exists():
        parser.error("Existing output refused; choose a fresh identity")
    if args.plan_only:
        print(json.dumps(plan_study(args.config, args.output), indent=2))
        return 0
    if args.case is None:
        parser.error("--case is required unless --plan-only")
    try:
        result = run_case(args.config, args.case, args.output)
    except Exception:
        if output.exists():
            save(output / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    print(json.dumps({"status": result["status"], "case": result["case"]["name"],
                      "checks": len(result["checks"]), "failed": result["failed_checks"],
                      "eligible_saved_updates": sum(r["measured_eligible"] for r in result["rows"])}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
