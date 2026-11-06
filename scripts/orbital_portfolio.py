import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["connected", "fixed", "feasible_fair"]
LABELS = ["Connected", "Fixed orbital", "Feasibility + fairness"]
COLORS = ["#64748b", "#2563eb", "#d97706"]


def collect(recovery, engineering_job):
    sources = {}

    def read(relative):
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    repeat = read("results/orbital/mission-repeats-v1/comparison.json")
    if repeat["incomplete"] or len(repeat["runs"]) != 9 or not repeat["distinct_seed_initializations"]:
        raise ValueError("Repeated study incomplete or initialization checks failed")
    if not all(c["same_initial_adapters"] for c in repeat["initialization_controls"].values()):
        raise ValueError("Initial adapters differ within paired controls")
    runs = []
    for entry in repeat["runs"]:
        state = read(entry["path"] + "/work_accounting.json")
        relative = entry["path"] + "/metrics.csv"
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:
            curve = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        runs.append({**entry, "accepted_per_client": state["accepted_updates"], "curve": curve})
    repeat.pop("codec_probe", None)
    geometry = read("results/orbital/engineering-v1/geometry.json")
    recovered = read(recovery + "/results.json")
    if geometry["status"] != "passed" or recovered["status"] != "passed":
        raise ValueError("Engineering studies did not pass")
    junit_path = f"results/orbital-accounting/{engineering_job}/tests.xml"
    junit = ROOT / junit_path
    sources[junit_path] = hashlib.sha256(junit.read_bytes()).hexdigest()
    import xml.etree.ElementTree as ET
    suites = ET.fromstring(junit.read_text())
    checks = {k: sum(int(s.attrib.get(k, 0)) for s in suites.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    if checks["failures"] or checks["errors"]:
        raise ValueError("Protocol checks failed")
    final = read("results/orbital/final-evaluation-22428040/results.json")
    quality = [{"label": label, "validation": final["results"][key]["validation"], "test": final["results"][key]["test"], "ifeval": final["results"][key]["ifeval"]} for label, key in
               (("Base", "base"), ("Orbital, epoch 2", "results/orbital/epoch-22406221"), ("Connected, epoch 2", "results/orbital/epoch-22428039"))]
    final_study = read("results/orbital/final-study-v1/study.json")
    first = read("results/orbital/mission-case62-v1/comparison.json")
    codec = read("results/orbital/codec-clients-22453177/results.json")
    contacts = read("results/orbital/mission-repeat-v1-s42-fixed/contacts.json")
    timelines = {}
    for policy in ("fixed", "feasible_fair"):
        run_path = "results/orbital/mission-repeat-v1-s42-" + policy
        relative = run_path + "/transfers.jsonl"
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipts = [json.loads(line) for line in path.read_text().splitlines()]
        timelines[policy] = {"transfers": [r for r in receipts if r["id"].startswith("r0-")], "plan": read(run_path + "/round-0000/plan.json")}
    compressed = read("results/orbital/compressed-feasibility-v1/comparison.json")
    costs = dict(final_study["allocated_gpu_hours_by_role"])
    costs["initial_mission_comparison"] = first["training_array_gpu_hours"]
    costs["nine_repeat_runs"] = repeat["training_array_gpu_hours"]
    return {"schema_version": 1, "repeat": repeat, "runs": runs, "quality": quality, "codec": codec,
            "contacts": contacts, "timelines": timelines, "compressed": compressed,
            "gpu_hours": costs, "geometry": geometry, "recovery": recovered, "protocol_checks": checks,
            "engineering_job": str(engineering_job), "source_sha256": sources,
            "scope": "Archived GPU studies plus independent CPU engineering additions. No new training or benchmark selection."}


def make_figures(data):
    figures = []

    def start(name, title, subtitle, panels=2, height=4.6):
        fig, axes = plt.subplots(1, panels, figsize=(12, height), squeeze=False)
        fig.suptitle(title, x=0.065, ha="left", fontsize=17, fontweight="bold", y=0.99)
        fig.text(0.065, 0.90, subtitle, fontsize=9.5, color="#475569")
        fig.subplots_adjust(top=0.79, bottom=0.18, left=0.075, right=0.97, wspace=0.35)
        figures.append((name, fig, title, subtitle))
        return axes[0]

    def bars(ax, values, ylabel, colors=COLORS, labels=LABELS):
        result = ax.bar(np.arange(len(values)), values, color=colors, width=0.62)
        ax.set_xticks(np.arange(len(values)), labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.bar_label(result, fmt="%.2f", padding=4, fontsize=9)
        ax.set_ylim(0, max(values) * 1.2)
        return result

    runs = data["runs"]
    by_policy = [[r for r in runs if r["policy"] == p] for p in POLICIES]
    ax = start("01_mission_tradeoffs", "More accepted work comes with a quality tradeoff", "Real five-GPU runs · three paired training seeds · one synthetic 24-hour mission · full validation: 344 conversations")
    for i, group in enumerate(by_policy):
        y = [r["validation_loss"] for r in group]
        ax[0].errorbar(i, np.mean(y), yerr=np.std(y, ddof=1), fmt="o", color=COLORS[i], capsize=7, markersize=8)
        ax[0].scatter(np.full(3, i) + [-0.07, 0, 0.07], y, color=COLORS[i], s=23, alpha=0.65)
    ax[0].set_xticks(range(3), LABELS, fontsize=9)
    ax[0].set_ylabel("Full validation answer loss ↓")
    ax[0].set_title("Mean ± sample standard deviation; dots = seeds", fontsize=10)
    accepted = [np.mean([r["accepted_steps"] for r in g]) for g in by_policy]
    wasted = [np.mean([r["performed_steps"] - r["accepted_steps"] for r in g]) for g in by_policy]
    ax[1].bar(range(3), accepted, color=COLORS, label="Accepted into global model")
    ax[1].bar(range(3), wasted, bottom=accepted, color="#fecaca", hatch="//", label="Performed, not accepted")
    for i, (a, w) in enumerate(zip(accepted, wasted)):
        ax[1].text(i, a + w + 3, f"{a:.0f} accepted\n{a + w:.0f} performed", ha="center", fontsize=8)
    ax[1].set_xticks(range(3), LABELS, fontsize=9)
    ax[1].set_ylabel("Optimizer steps per mission ↑")
    ax[1].set_ylim(0, 145)
    ax[1].legend(fontsize=8)

    ax = start("02_mission_learning", "Learning and delivered work over the simulated mission", "Solid line = three-seed mean; shaded band = sample standard deviation · intermediate loss uses a fixed 32-conversation development slice")
    for i, group in enumerate(by_policy):
        for axis, field, scale in ((ax[0], "validation_answer_loss", 1), (ax[1], "accepted_steps", 1)):
            values = np.array([[r[field] / scale for r in g["curve"]] for g in group])
            hours = np.array([r["mission_seconds"] / 3600 for r in group[0]["curve"]])
            axis.plot(hours, values.mean(axis=0), "o-", color=COLORS[i], label=LABELS[i])
            axis.fill_between(hours, values.mean(axis=0) - values.std(axis=0, ddof=1), values.mean(axis=0) + values.std(axis=0, ddof=1), color=COLORS[i], alpha=0.15)
            axis.set_xlabel("Simulated mission hours")
    ax[0].set_ylabel("Development answer loss, 32 examples ↓")
    ax[1].set_ylabel("Cumulative accepted optimizer steps ↑")
    ax[0].legend(fontsize=8)

    ax = start("03_contact_schedule", "Orbital visibility constrains the training schedule", "Seed 42 · first four simulated hours · one hypothetical ground station · 1 Mbps · assumed 60 seconds per optimizer step", height=5)
    for axis, policy in zip(ax, ("fixed", "feasible_fair")):
        for c in data["contacts"]["contacts"]:
            if c["start"] < 14400:
                axis.broken_barh([(c["start"] / 3600, (min(c["end"], 14400) - c["start"]) / 3600)], (c["client"] - 0.32, 0.64), facecolor="#e2e8f0")
        timeline = data["timelines"][policy]
        for receipt in timeline["transfers"]:
            color = "#2563eb" if receipt["direction"] == "download" else "#b91c1c"
            for chunk in receipt["chunks"]:
                axis.broken_barh([(chunk["start"] / 3600, (chunk["finish"] - chunk["start"]) / 3600)], (receipt["client"] - 0.18, 0.36), facecolor=color)
        for task in timeline["plan"]["tasks"]:
            axis.broken_barh([(task["ready"] / 3600, (task["complete"] - task["ready"]) / 3600)], (task["client"] - 0.10, 0.20), facecolor="#d97706")
        axis.set(xlim=(0, 4), yticks=range(5), ylabel="Virtual client", xlabel="Simulated mission hours")
        axis.set_title(LABELS[POLICIES.index(policy)], fontsize=11)
    fig = ax[0].figure
    fig.legend(handles=[Patch(color=c, label=l) for c, l in (("#e2e8f0", "Visible"), ("#2563eb", "Download"), ("#d97706", "Compute assumption"), ("#b91c1c", "Upload"))], loc="lower center", ncol=4, fontsize=8)

    ax = start("04_participation_and_bytes", "Participation and communication cost", "Accepted update counts are identical across these three seeds because geometry and scheduling are deterministic")
    x = np.arange(5)
    for i, group in enumerate(by_policy):
        counts = np.mean([r["accepted_per_client"] for r in group], axis=0)
        ax[0].bar(x + (i - 1) * 0.24, counts, width=0.24, label=LABELS[i], color=COLORS[i])
    ax[0].set(xlabel="Virtual client", ylabel="Accepted updates per 24-hour mission", xticks=x)
    ax[0].legend(fontsize=8)
    bars(ax[1], [np.mean([r["wire_bytes"] for r in g]) / 2**20 for g in by_policy], "Simulated wire traffic (MiB) ↓")

    ax = start("05_heldout_quality", "Conversation fit improves; instruction following does not", "Separate full two-epoch study · all 365 held-out test conversations · fixed 100/541 IFEval sample, strict prompt scoring")
    labels = [r["label"] for r in data["quality"]]
    bars(ax[0], [r["test"]["answer_loss"] for r in data["quality"]], "Token-weighted test answer loss ↓", labels=labels)
    rates = [r["ifeval"]["strict"]["prompt_accuracy"] for r in data["quality"]]
    bars(ax[1], [100 * r for r in rates], "IFEval strict prompt pass rate (%) ↑", labels=labels)
    ax[1].set_ylim(0, 105)
    ax[1].set_title("Base 80/100; orbital 77/100; connected 76/100", fontsize=10)

    ax = start("06_codec_study", "Int8 updates reduce serialized size in the CPU codec study", "All five archived clients · two sequential epoch deltas · numerical codec study; no compressed GPU learning-quality claim")
    values = data["codec"]["results"]
    raw = np.mean([v["raw_serialized_bytes"] for v in values]) / 2**20
    quant = np.mean([v["int8_serialized_bytes"] for v in values]) / 2**20
    bars(ax[0], [raw, quant], "Mean serialized update size (MiB) ↓", colors=["#64748b", "#0f766e"], labels=["Float32 B state", "Int8 delta + metadata"])
    second = [r for r in values if r["epoch"] == 2]
    for offset, key, label, color in ((-0.18, "cumulative_error_no_feedback", "Without error feedback", "#64748b"), (0.18, "cumulative_error_feedback", "With error feedback", "#0f766e")):
        ax[1].bar(np.arange(5) + offset, [r[key] for r in second], width=0.35, label=label, color=color)
    ax[1].set(xlabel="Virtual client", ylabel="Epoch-two cumulative relative L2 error ↓", xticks=range(5))
    ax[1].legend(fontsize=8)

    ax = start("07_gpu_costs", "Actual allocation cost is separate from simulated mission time", "Enumerated completed studies only; includes allocated loading, evaluation and idle time · excludes unrelated jobs and later CPU engineering checks", panels=1, height=5.8)[0]
    costs = data["gpu_hours"]
    selected = [(k.replace("_", " "), v) for k, v in costs.items() if v > 0]
    y = np.arange(len(selected))
    result = ax.barh(y, [v for k, v in selected], color="#2563eb")
    ax.set_yticks(y, [k for k, v in selected], fontsize=9)
    ax.bar_label(result, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlabel(f"Allocated GPU hours — selected scopes total {sum(costs.values()):.2f}")
    ax.invert_yaxis()
    ax.figure.subplots_adjust(left=0.24, top=0.78, bottom=0.13)
    ax.set_xlim(0, max(v for k, v in selected) * 1.2)

    ax = start("08_compression_sensitivity", "Compression forecasts depend on the mission budget", "CPU planning only · one ground station · 60 simulated seconds per step · 24-hour mission · cells show added accepted steps from smaller payloads")
    for axis, policy in zip(ax, ("fixed", "feasible_fair")):
        cases = [r for r in data["compressed"][policy] if r["orbital_stations"] == 1 and r["step_seconds"] == 60]
        rates = sorted({r["rate_bps"] for r in cases})
        rounds = sorted({r["round_seconds"] for r in cases})
        values = np.array([[next(r["int8_planned_accepted_steps"] - r["raw_accepted_steps"] for r in cases if r["rate_bps"] == rate and r["round_seconds"] == length) for rate in rates] for length in rounds])
        vmax = max(1, np.max(np.abs(values)))
        image = axis.imshow(values, cmap="PuOr", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(rates)), [f"{r / 1e6:g}" for r in rates])
        axis.set_yticks(range(len(rounds)), [f"{r / 3600:g}" for r in rounds])
        axis.set(xlabel="Assumed link rate (Mbps)", ylabel="Round deadline (hours)")
        axis.set_title(LABELS[POLICIES.index(policy)], fontsize=11)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                axis.text(col, row, f"{values[row, col]:+d}", ha="center", va="center", fontsize=10, color="black" if abs(values[row, col]) < vmax * 0.55 else "white")

    ax = start("09_geometry_validation", "A second coordinate pipeline checks synthetic contact timing", "Shared SGP4 propagation · Astropy frame transforms and grid/bisection search versus archived Skyfield contacts · numerical validation only")
    checks = data["geometry"]["checks"]
    bars(ax[0], [c["max_boundary_difference_seconds"] for c in checks], "Maximum contact boundary difference (seconds) ↓", colors=["#0f766e"] * len(checks), labels=[f"Client {c['client']}" for c in checks])
    ax[0].axhline(2, color="#b91c1c", linestyle="--", label="Predeclared 2-second tolerance")
    ax[0].set_ylim(0, max(2.3, ax[0].get_ylim()[1]))
    ax[0].legend(fontsize=8)
    for client in range(5):
        values = [s for s in data["geometry"]["samples"] if s["client"] == client]
        ax[1].plot([s["seconds"] / 3600 for s in values], [s["astropy_elevation_deg"] - s["skyfield_elevation_deg"] for s in values], linewidth=1, label=f"Client {client}")
    ax[1].set(xlabel="Simulated mission hours", ylabel="Astropy − Skyfield elevation (degrees)")
    ax[1].legend(fontsize=7, ncol=3)

    ax = start("10_recovery_demonstration", "Verified chunks survive a receiver process interruption", "Real saved adapter bytes · forced process exit after commit/before acknowledgment · separate CPU extension, not archived GPU training")
    trace = data["recovery"]["trace"]
    x = np.arange(len(trace))
    ax[0].plot(x, [t["attempted_payload_bytes"] / 2**20 for t in trace], "o-", color="#64748b", label="Attempted payload, including retries")
    ax[0].plot(x, [t["verified_bytes"] / 2**20 for t in trace], "o-", color="#0f766e", label="Unique verified payload")
    ax[0].set(xlabel="Recorded fault/recovery event index", ylabel="Payload volume (MiB)")
    ax[0].legend(fontsize=8)
    ax[1].step(x, [t["eligible_count"] for t in trace], where="post", color="#2563eb", linewidth=2)
    ax[1].set(xlabel="Recorded fault/recovery event index", ylabel="Transactional eligible-update count", ylim=(-0.1, 1.3), yticks=[0, 1])
    ax[1].text(0.03, 0.88, "Corrupt / partial payload: rejected\nDuplicate chunks: no double counting\nComplete verified payload: one eligibility record", transform=ax[1].transAxes, fontsize=9, va="top")
    return figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-bundle")
    parser.add_argument("--recovery")
    parser.add_argument("--engineering-job")
    parser.add_argument("--output", default="docs/orbital_sft/figures")
    args = parser.parse_args()
    if args.from_bundle:
        data = json.loads((ROOT / args.from_bundle).read_text())
    else:
        if not args.recovery or not args.engineering_job:
            parser.error("Supply --from-bundle or both --recovery and --engineering-job")
        data = collect(args.recovery, args.engineering_job)
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay in project")
    output.mkdir(parents=True, exist_ok=True)
    evidence = output.parent / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    bundle = evidence / "plot_data.json"
    bundle.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                         "grid.alpha": 0.18, "savefig.facecolor": "white", "svg.hashsalt": "orbital-v1"})
    figures = make_figures(data)
    gallery = ["# Measured results and engineering gallery", "", "All graphics are generated from the [portable evidence bundle](../evidence/plot_data.json). PNG is for previews; SVG and PDF are available for export. The combined [PDF](orbital_results.pdf) includes all figures.", ""]
    outputs = {}
    with PdfPages(output / "orbital_results.pdf") as pdf:
        for name, fig, title, subtitle in figures:
            for extension in ("png", "svg", "pdf"):
                path = output / f"{name}.{extension}"
                fig.savefig(path, dpi=180, bbox_inches="tight")
                outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            pdf.savefig(fig, bbox_inches="tight")
            gallery.extend([f"## {title}", "", subtitle + ".", "", f"![{title}]({name}.png)", "", f"[SVG]({name}.svg) · [PDF]({name}.pdf)", ""])
            plt.close(fig)
    outputs["orbital_results.pdf"] = hashlib.sha256((output / "orbital_results.pdf").read_bytes()).hexdigest()
    (output / "README.md").write_text("\n".join(gallery))
    with (evidence / "mission_runs.csv").open("w") as stream:
        fields = ["seed", "policy", "validation_loss", "performed_steps", "accepted_steps", "accepted_updates", "wire_bytes"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["runs"])
    snapshot = {"plot_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(), "source_sha256": data["source_sha256"],
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {p: version(p) for p in ("matplotlib", "numpy", "astropy", "skyfield", "sgp4")},
                "outputs_sha256": outputs, "protocol_checks": data["protocol_checks"]}
    (evidence / "manifest.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False))
    (evidence / "README.md").write_text("# Portable evidence\n\n`plot_data.json` contains the plotted measurements, engineering results, and SHA256 identities of original inputs. `mission_runs.csv` provides all nine measured outcomes. `manifest.json` records input/output hashes and plotting versions. No weights, conversations, tokens or credentials are included.\n\nOriginal artifacts remain under the project-local `results/` directory, which is ignored by Git. This compact bundle and the gallery can be included in a reviewed repository.\n")
    print(f"Generated {len(figures)} figures in PNG/SVG/PDF, a combined PDF, gallery and portable evidence at {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
