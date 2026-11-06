import argparse
import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array-job", required=True)
    parser.add_argument("--tests-job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must stay inside the project")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/orbital_sft/contact_transport.json").read_text())
    results, rows, missing, provenance = [], [], [], []
    prediction_checks = {}
    for index, case in enumerate(config["cases"]):
        directory = output / f"case-{index}"
        path = directory / "results.json"
        if not path.is_file():
            missing.append(index)
            continue
        result = json.loads(path.read_text())
        result["index"] = index
        results.append(result)
        provenance.append(json.loads((directory / "provenance.json").read_text()))
        before = json.loads((directory / "prediction-before-run.json").read_text())
        executed = json.loads((directory / "execution/prediction.json").read_text())
        gate = ROOT / f"results/orbital/contact-plan-{args.tests_job}/prediction-{index}.json"
        prediction_checks[f"case-{index}:frozen_predictions"] = before == executed and gate.is_file() and before == json.loads(gate.read_text())
        for row in result["rows"]:
            rows.append({"case_index": index, "case": case["name"], **row})
    checks = {"all_ten_cases_present": not missing,
              "all_case_checks_passed": len(results) == len(config["cases"]) and all(r["status"] == "passed" for r in results),
              "same_source_and_configuration": bool(provenance) and all(p["source_sha256"] == provenance[0]["source_sha256"] for p in provenance),
              "same_saved_payload": bool(provenance) and len({p["input_sha256"] for p in provenance}) == 1,
              "stress_exercises_contention": any(r["case"].get("co_visible_stress") and r["contention_slots"] > 0 and r["executed_transactions"] > 0 for r in results)}
    checks.update(prediction_checks)
    study_path = ROOT / "configs/orbital_sft/contact_transport_study.json"
    study = json.loads(study_path.read_text()) if study_path.exists() else {}
    if study.get("source_sha256"):
        checks["source_matches_submission"] = all(p["source_sha256"] == study["source_sha256"] for p in provenance)
    xml = ROOT / f"results/orbital-accounting/{args.tests_job}/contact-tests.xml"
    tests = None
    if xml.exists():
        tree = ET.parse(xml)
        suites = list(tree.iter("testsuite"))
        tests = {name: sum(int(s.get(name, 0)) for s in suites) for name in ("tests", "failures", "errors", "skipped")}
        checks["cpu_tests_passed"] = tests["tests"] > 0 and tests["failures"] == tests["errors"] == tests["skipped"] == 0
    else:
        checks["cpu_tests_passed"] = False
    accounting = subprocess.run(["sacct", "-j", args.array_job, "-n", "-P", "--format=JobID,State,ElapsedRaw,AllocCPUS,AllocTRES,ExitCode"],
                                capture_output=True, text=True, check=False)
    (output / "accounting.psv").write_text(accounting.stdout + accounting.stderr)
    allocated_cpu_hours = 0
    task_records = []
    for line in accounting.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 6 and re.fullmatch(re.escape(args.array_job) + r"_\d+", fields[0]):
            task_records.append(fields)
            allocated_cpu_hours += int(fields[2]) * int(fields[3]) / 3600
    checks["ten_successful_cpu_tasks"] = len(task_records) == 10 and all(f[1] == "COMPLETED" and f[5] == "0:0" for f in task_records)
    checks["no_gpu_allocation"] = bool(task_records) and all("gpu" not in f[4].lower() for f in task_records)
    if rows:
        with (output / "clients.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status = "passed" if all(checks.values()) else "failed"
    limits = [
        "Actual TCP bytes are replayed on one CPU node; no new training, aggregation, flight hardware or onboard energy calibration.",
        "Three archived synthetic SGP4 geometries are uniformly compressed by 360:1: a 24-hour mission is replayed in at most 240 seconds.",
        "The shaped wall-clock rate is 64 Mbps, equivalent to 177777.78 bits per modeled mission second; this is a declared sensitivity, not a flight-radio measurement.",
        "Fixed 250 ms TDMA slots include a 150 ms processing guard. Station/client reservations are exclusive; unused allocations are not reassigned.",
        "Outages are synthetic slot-aligned failures, with two selected slot starts per station and client, each lasting one to three slots. Sub-slot outage behavior is not tested here.",
        "Original orbital contacts have no shared-station overlap. The tenth case deliberately clones one client's contact times across five clients to stress contention; it is not an independently propagated constellation.",
        "The proxy ingress is local and unshaped, buffering one bounded request per client. Actual proxy-to-coordinator sockets are opened only during contact slots and closed after each transaction.",
        "Byte accounting covers shaped application frames, excluding TCP/IP overhead and inbound buffering. Measured timing includes CPU/filesystem overhead.",
        "Predictions model the new slotted socket protocol, not the older LinkEngine. Shared contact inputs and code audits establish implementation consistency, not independent physical validation.",
        "Eligibility is durable acceptance of a saved adapter, separate from client confirmation and from a training aggregation. Acknowledgment loss can leave an eligible update unconfirmed.",
    ]
    summary = {"status": status, "checks": checks, "missing_cases": missing, "tests": tests,
               "array_job": args.array_job, "tests_job": args.tests_job,
               "allocated_cpu_hours_excluding_tests_and_report": allocated_cpu_hours,
               "case_results": [{"index": r["index"], "name": r["case"]["name"], "status": r["status"],
                                 "predicted_eligible": sum(c["predicted_eligible"] for c in r["rows"]),
                                 "measured_eligible": sum(c["measured_eligible"] for c in r["rows"]),
                                 "confirmed_clients": sum(c["measured_delivered"] for c in r["rows"]),
                                 "forwarded_bytes": sum(c["measured_request_bytes"] + c["measured_response_bytes"] for c in r["rows"]),
                                 "outage_slots": sum(c["outage_slots"] for c in r["rows"]),
                                 "checks": len(r["checks"]), "failed_checks": r["failed_checks"],
                                 "contention_slots": r["contention_slots"]} for r in results],
               "rows": rows, "limits": limits}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Orbital contact socket replay", "", f"Status: **{status}**", "",
             f"CPU array: {args.array_job}; tests: {args.tests_job}. Allocated CPU hours excluding tests/report: {allocated_cpu_hours:.4f}.", "",
             "| Case | Predicted eligible | Measured eligible | Confirmed clients | Forwarded MiB | Outage slots | Checks |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in summary["case_results"]:
        lines.append(f"| {r['name']} | {r['predicted_eligible']} | {r['measured_eligible']} | {r['confirmed_clients']} | {r['forwarded_bytes']/2**20:.3f} | {r['outage_slots']} | {r['status']} |")
    lines.extend(["", "## Failed or missing checks", "", *([f"- {k}" for k,v in checks.items() if not v] or ["None."]),
                  "", f"Missing cases: {missing}.", "", "## Limits", "", *[f"- {limit}" for limit in limits]])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "orbital-contact-v1"})
    artifacts = []
    with PdfPages(destination / "contact_results.pdf") as pdf:
        def publish(fig, name):
            for suffix in ("png", "svg", "pdf"):
                path = destination / f"{name}.{suffix}"
                fig.savefig(path, dpi=160, bbox_inches="tight")
                artifacts.append(path)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not results:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            ax.text(0.05, 0.5, f"No completed case results. Missing cases: {missing}", transform=ax.transAxes)
            publish(fig, "00_incomplete_results")
        if results:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
            cases = summary["case_results"]
            positions = list(range(len(cases)))
            axes[0].bar([p-0.16 for p in positions], [r["predicted_eligible"] for r in cases], width=0.3, label="Predicted", color="#93c5fd")
            axes[0].bar([p+0.16 for p in positions], [r["measured_eligible"] for r in cases], width=0.3, label="Measured", color="#2563eb")
            axes[0].set(xticks=positions, xticklabels=[str(r["index"]) for r in cases], xlabel="Frozen case index; 9 = co-visible stress", ylabel="Eligible saved updates / 5", ylim=(0,5.6))
            axes[0].legend()
            for geometry, color in ((0,"#2563eb"),(3,"#d97706"),(4,"#059669")):
                group = [row for row in rows if config["cases"][row["case_index"]]["geometry"] == geometry and row["measured_seconds"] is not None and row["predicted_seconds"] is not None]
                axes[1].scatter([r["predicted_seconds"] for r in group], [r["measured_seconds"] for r in group], label=f"Geometry {geometry}", color=color, alpha=0.7)
            axes[1].plot([0,240],[0,240],"--",color="#94a3b8")
            axes[1].set(xlabel="Predicted confirmation (wall s)", ylabel="Measured confirmation (wall s)")
            axes[1].legend()
            fig.suptitle(f"Actual socket transfers under compressed orbital contacts — {status}")
            publish(fig, "01_delivery_and_timing")
        stress = next((r for r in results if r["case"].get("co_visible_stress")), None)
        if stress:
            directory = output / f"case-{stress['index']}/execution"
            manifest = json.loads((directory / "manifest.json").read_text())
            proxy = json.loads((directory / "proxy.json").read_text())
            first = min(c["start"] for c in manifest["contacts"])
            end = min(c["end"] for c in manifest["contacts"] if c["start"] == first)
            fig, ax = plt.subplots(figsize=(12,5), layout="constrained")
            palette = ["#2563eb","#d97706","#059669","#9333ea","#dc2626"]
            for client in range(5):
                ax.broken_barh([(first,end-first)], (client-.35,.7), facecolor="#e2e8f0")
            for slot in manifest["slots"]:
                if first <= slot["start"] < end:
                    ax.broken_barh([(slot["start"],slot["end"]-slot["start"])], (slot["client"]-.28,.56), facecolor=palette[slot["client"]], alpha=.45)
            for event in proxy["events"]:
                if first <= event.get("start",-1) < end:
                    if "link_close" in event:
                        ax.broken_barh([(event["link_open"], event["link_close"]-event["link_open"])], (event["client"]-.12,.24), facecolor=palette[event["client"]])
                    elif event.get("outcome") == "outage":
                        ax.scatter(event["start"],event["client"],marker="x",color="black")
            ax.set(yticks=range(5), ylabel="Client", xlabel="Compressed wall-clock seconds", xlim=(first-.05,end+.05),
                   title="Co-visible stress: allocated TDMA slots and actual socket intervals")
            fig.supxlabel("Cloned visibility stress, not independent orbital propagation · pale = visibility · medium = allocation · dark = actual socket", fontsize=9)
            publish(fig, "02_shared_station_stress")
    artifacts.append(destination / "contact_results.pdf")
    (destination / "manifest.json").write_text(json.dumps({"input_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
        "files": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}, indent=2) + "\n")
    if study_path.exists() and str(study.get("array_job")) == args.array_job:
        study.update(status=status, report=str((output / "report.md").relative_to(ROOT)),
                     allocated_cpu_hours_excluding_tests_and_report=allocated_cpu_hours)
        study_path.write_text(json.dumps(study, indent=2) + "\n")
    (ROOT / "docs/orbital_sft/evidence/contact_transport_summary.json").write_bytes((output / "summary.json").read_bytes())
    (ROOT / "docs/orbital_sft/evidence/contact_transport_report.md").write_bytes((output / "report.md").read_bytes())
    print(json.dumps({"status": status, "missing": missing, "failed": [k for k,v in checks.items() if not v], "report": str(output / "report.md")}, indent=2))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
