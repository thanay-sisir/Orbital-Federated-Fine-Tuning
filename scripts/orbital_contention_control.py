import argparse
import hashlib
import json
import os
import traceback
import zipfile

from swarmshield.orbital.contact_transport import run_case
from swarmshield.orbital.contention_validation import validate_payload_progress
from swarmshield.orbital.socket_transport import ROOT, project_path, save


def main():
    parser = argparse.ArgumentParser(description="CPU-only no-outage shared-station adapter control")
    parser.add_argument("--config", default="configs/orbital_sft/contention_control.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use the CPU Slurm job")
    output = project_path(args.output)
    if output.exists():
        parser.error("Existing output refused")
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text())
    paths = ["src/swarmshield/orbital/contention_validation.py", "scripts/orbital_contention_control.py",
             "scripts/orbital_contention_plots.py", "tests/orbital_sft/test_contention_validation.py",
             "infra/rivanna/contention_control.sbatch", "infra/rivanna/common.sh", args.config]
    sources = {name: project_path(name).read_bytes() for name in paths}
    try:
        base = run_case(args.config, 0, args.output)
        validation = validate_payload_progress(output / "execution", project_path(config["payload"]),
                      config["minimum_clients_with_verified_payload"], config["minimum_verified_bytes_per_client"])
        current = json.loads((output / "execution/manifest.json").read_text())
        reference_path = project_path(config["reference_execution"]) / "manifest.json"
        reference = json.loads(reference_path.read_text())
        checks = {"contact_protocol_checks": base["status"] == "passed", **validation["checks"],
                  "same_nominal_contacts_as_previous_stress": current["contacts"] == reference["contacts"],
                  "same_nominal_slots_as_previous_stress": current["slots"] == reference["slots"],
                  "same_payload_as_previous_stress": current["transfers"] == reference["transfers"],
                  "same_radio_and_storage_settings": current["network"] == reference["network"] and current["queue_bytes"] == reference["queue_bytes"]}
        result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                  "payload_validation": validation, "case_checks": len(base["checks"]),
                  "eligibility": [{"client": r["client"], "eligible": r["measured_eligible"]} for r in base["rows"]],
                  "reference_execution": config["reference_execution"],
                  "limitations": ["One-host TCP replay with cloned co-visible windows; no independently propagated co-visible constellation.",
                                  "Five clients replay the same archived adapter. This experiment performs no new training or aggregation.",
                                  "The short shared-station budget tests partial-payload progress, not complete adapter delivery.",
                                  "Rate, TDMA processing guard and 360:1 mission-time compression match the preceding study; they are not calibrated flight parameters.",
                                  "The Jain index describes byte shares in this one fixed schedule; it does not establish fairness optimality."]}
        save(output / "payload_validation.json", result)
        with zipfile.ZipFile(output / "validation_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sources.items():
                archive.writestr(name, data)
        save(output / "validation_source_hashes.json", {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()})
        lines = ["# Shared-station payload control", "", f"Status: **{result['status']}**.", "",
                 f"{base['status']} on {len(base['checks'])} protocol checks; {sum(checks.values())}/{len(checks)} payload/control checks passed.", "",
                 "The previous co-visible stress is held fixed except that outages are removed. Verified bytes below come from read-only inspection of receiver chunks, compared byte-for-byte and by SHA256 with the archived adapter.", "",
                 "| Client | Predicted verified chunks | Measured verified chunks | Verified payload bytes |", "|---|---:|---:|---:|"]
        for row in validation["clients"]:
            lines.append(f"| {row['client']} | {row['predicted_verified_chunks']} | {row['measured_verified_chunks']} | {row['measured_verified_payload_bytes']} |")
        lines.extend(["", f"Total verified payload: {validation['total_verified_payload_bytes']} bytes. Jain byte-share index: {validation['jain_verified_byte_share']:.6f}.", "",
                      f"Clients meeting the {validation['minimum_bytes_per_client']}-byte requirement: {validation['clients_meeting_payload_threshold']}/{validation['minimum_clients']}.", "",
                      "## Failed checks", "", *([f"- {k}" for k,v in checks.items() if not v] or ["None."]),
                      "", "## Limits", "", *[f"- {limit}" for limit in result["limitations"]]])
        (output / "payload_report.md").write_text("\n".join(lines) + "\n")
    except Exception:
        if output.exists():
            save(output / "failure.json", {"traceback": traceback.format_exc()})
        raise
    print(json.dumps({"status": result["status"], "total_verified_payload_bytes": validation["total_verified_payload_bytes"],
                      "clients_with_payload": validation["clients_meeting_payload_threshold"], "report": str(output / "payload_report.md")}, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
