import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("fixed", "retry_backoff", "feasible_fair")
LABELS = ("Fixed", "Retry + backoff", "Feasibility + fairness")
COLORS = ("#2563eb", "#64748b", "#d97706")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-bundle")
    args = parser.parse_args()
    output = ROOT / "docs/orbital_sft/outage_figures"
    output.mkdir(parents=True, exist_ok=True)
    if args.from_bundle:
        data = json.loads((ROOT / args.from_bundle).read_text())
    else:
        source = ROOT / "results/orbital/outages-v1"
        cpu = json.loads((source / "cpu_summary.json").read_text())
        selected = cpu["selected"]
        geometry = json.loads((source / f"cpu/geometry-{selected['geometry']}/contacts.json").read_text())
        details = json.loads((ROOT / selected["details"]).read_text())
        trace = json.loads((ROOT / selected["trace"]).read_text())
        checks = [json.loads((source / f"geometry-check-{i}.json").read_text()) for i in range(6)]
        data = {"rows": cpu["rows"], "selected": selected, "calibration": cpu["calibration"],
                "energy_scope": cpu["energy_scope"], "geometry": geometry, "details": details, "trace": trace,
                "geometry_checks": [{"status": c["status"], "checks": c["checks"], "versions": c["versions"]} for c in checks],
                "scope": cpu["limits"]}
        gpu_path = source / "gpu_results.json"
        if gpu_path.exists():
            data["gpu"] = json.loads(gpu_path.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.18,
        "savefig.facecolor": "white", "svg.hashsalt": "orbital-outages-v1"})
    figures = []

    def make(name, title, subtitle, panels=2, height=5):
        fig, axes = plt.subplots(1, panels, figsize=(12, height), squeeze=False)
        fig.suptitle(title, x=0.065, ha="left", fontsize=16, fontweight="bold", y=0.99)
        fig.text(0.065, 0.90, subtitle, fontsize=9, color="#475569")
        fig.subplots_adjust(top=0.78, bottom=0.2, left=0.08, right=0.97, wspace=0.35)
        figures.append((name, title, subtitle, fig))
        return axes[0]

    rows = data["rows"]
    baselines = {(r["geometry"], r["rate_bps"], r["step_seconds"], r["policy"]): r["accepted_steps"] for r in rows if r["profile"] == "none" and r["fault_seed"] == 101}
    ax = make("01_outage_delivery", "Useful training work under unexpected outages", "648 audited CPU trajectories · six geometries · three fault seeds · radio and compute sensitivity · planned work, not learned quality")
    for i, policy in enumerate(POLICIES):
        boxes = []
        for severity in ("none", "low", "high"):
            sample = [r for r in rows if r["policy"] == policy and r["profile"] == severity and (severity != "none" or r["fault_seed"] == 101)]
            ratios = [r["accepted_steps"] / baselines[(r["geometry"], r["rate_bps"], r["step_seconds"], policy)] for r in sample if baselines[(r["geometry"], r["rate_bps"], r["step_seconds"], policy)] > 0]
            boxes.append(ratios)
        bp = ax[0].boxplot(boxes, positions=np.arange(3) + (i - 1) * 0.22, widths=0.19, patch_artist=True, manage_ticks=False)
        for b in bp["boxes"]:
            b.set_facecolor(COLORS[i])
            b.set_alpha(0.6)
        means = [np.mean([r["accepted_steps"] for r in rows if r["policy"] == policy and r["profile"] == severity]) for severity in ("none", "low", "high")]
        ax[1].plot(range(3), means, "o-", color=COLORS[i], label=LABELS[i])
    ax[0].set(xticks=range(3), xticklabels=["None", "Low", "High"], ylabel="Accepted steps / own no-outage control", xlabel="Synthetic outage profile")
    ax[1].set(xticks=range(3), xticklabels=["None", "Low", "High"], ylabel="Mean accepted optimizer steps", xlabel="Synthetic outage profile")
    ax[1].legend(fontsize=8)

    ax = make("02_outage_costs", "Failed transfers and interrupted compute have explicit costs", "Failure bytes count only modeled transmitted prefixes; unfinished compute is a simulated cost, not GPU-executed partial gradients")
    for i, policy in enumerate(POLICIES):
        for axis, field, scale, label in ((ax[0], "failed_wire_bytes", 2**20, "Failed-attempt wire traffic (MiB)"), (ax[1], "wasted_compute_seconds", 60, "Discarded partial-step compute (minutes)")):
            values = [np.mean([r[field] / scale for r in rows if r["policy"] == policy and r["profile"] == severity]) for severity in ("none", "low", "high")]
            axis.bar(np.arange(3) + (i - 1) * 0.24, values, width=0.23, color=COLORS[i], label=LABELS[i])
            axis.set(xticks=range(3), xticklabels=["None", "Low", "High"], ylabel=label, xlabel="Synthetic outage profile")
    ax[0].legend(fontsize=8)

    ax = make("03_geometry_sensitivity", "Delivery depends on geometry as well as policy", "High-outage profile · means over fault seed, radio rate and compute assumptions · numerical geometry cross-checks retained", panels=1)[0]
    values = np.array([[np.mean([r["accepted_steps"] for r in rows if r["geometry"] == g and r["profile"] == "high" and r["policy"] == p]) for g in range(6)] for p in POLICIES])
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(3), LABELS)
    ax.set_xticks(range(6), [f"{alt} km\n{sites} site(s)" for alt in (450, 550, 700) for sites in (1, 2)])
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", color="white" if values[i,j] > (values.max() + values.min()) / 2 else "black")
    ax.figure.colorbar(image, ax=ax, label="Mean accepted optimizer steps")
    ax.grid(False)

    ax = make("04_realized_timeline", "An outage can consume the only usable contact window", "Selected fixed-policy CPU trace · first four mission hours · events are ground truth for analysis, not future information available to the policy", panels=1, height=5.7)[0]
    for contact in data["geometry"]["contacts"]:
        if contact["start"] < 14400:
            ax.broken_barh([(contact["start"] / 3600, (min(contact["end"], 14400) - contact["start"]) / 3600)], (contact["client"] - 0.34, 0.68), facecolor="#e2e8f0")
    for attempt in data["details"]["attempts"]:
        if attempt["start"] < 14400:
            ax.broken_barh([(attempt["start"] / 3600, (min(attempt["finish"], 14400) - attempt["start"]) / 3600)], (attempt["client"] - 0.17, 0.34), facecolor="#0f766e" if attempt["success"] else "#b91c1c")
    for event in data["trace"]["events"]:
        if event["start"] < 14400:
            targets = range(5) if event["kind"] == "station" else [event["client"]]
            for client in targets:
                ax.broken_barh([(event["start"] / 3600, (min(event["end"], 14400) - event["start"]) / 3600)], (client - 0.31, 0.62), facecolor="#fecaca", alpha=0.32)
    ax.set(xlim=(0, 4), ylim=(-0.6, 4.6), yticks=range(5), ylabel="Virtual client", xlabel="Simulated mission hours")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in (("#e2e8f0", "Nominal visibility"), ("#0f766e", "Successful attempt"), ("#b91c1c", "Failed attempt"), ("#fecaca", "Realized outage"))], fontsize=8, ncol=4, loc="upper center")

    ax = make("05_assumptions_and_energy", "Compute and power assumptions remain explicit", "Earth A100 timing is a reference, not onboard calibration · energy is a posthoc sensitivity calculation, not battery feasibility")
    observed = data["calibration"]["earth_a100_median_seconds_per_step"]
    names, values = (["Measured Earth A100\nreference", "Assumed profile A", "Assumed profile B"], [observed, 30, 60]) if observed is not None else (["Assumed profile A", "Assumed profile B"], [30, 60])
    bars = ax[0].bar(range(len(values)), values, color=["#64748b", "#d97706", "#d97706"][:len(values)])
    ax[0].bar_label(bars, fmt="%.1f", padding=3)
    ax[0].set(xticks=range(len(names)), xticklabels=names, ylabel="Seconds per optimizer step")
    selected = [r for r in rows if r["case"] == data["selected"]["case"]]
    for i, key in enumerate(("low_assumed_power", "high_assumed_power")):
        ax[1].bar(np.arange(3) + (i - 0.5) * 0.35, [next(r["energy_sensitivity_wh"][key] for r in selected if r["policy"] == p) for p in POLICIES], width=0.34, label=key.replace("_", " "), color=("#0f766e", "#64748b")[i])
    ax[1].set(xticks=range(3), xticklabels=LABELS, ylabel="Assumed client-active energy (Wh)")
    ax[1].legend(fontsize=8)

    if "gpu" in data:
        gpu = data["gpu"]
        ax = make("06_gpu_quality_and_work", "Actual GPU training: quality and delivered work", "One selected scenario, one training seed · all 344 validation conversations · conformance checks must pass; no broad quality claim")
        names = [r["label"].replace("_", "\n") for r in gpu["results"]]
        for axis, key, label in ((ax[0], "validation_loss", "Full validation answer loss ↓"), (ax[1], "accepted_steps", "Actual accepted optimizer steps ↑")):
            bars = axis.bar(range(len(names)), [r[key] for r in gpu["results"]], color=["#0f766e", *COLORS])
            axis.bar_label(bars, fmt="%.4f" if key == "validation_loss" else "%.0f", padding=3, fontsize=9)
            axis.set(xticks=range(len(names)), xticklabels=names, ylabel=label)
            axis.set_ylim(0, max([r[key] for r in gpu["results"]], default=1) * 1.2)
    evidence = ROOT / "docs/orbital_sft/evidence"
    bundle_path = evidence / "outages_plot_data.json"
    bundle_path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    gallery = ["# Mission-time outage study figures", "", "CPU graphics show modeled work, not learned quality. GPU quality appears only after actual runs and conformance reporting. [Portable inputs](../evidence/outages_plot_data.json).", ""]
    outputs = {}
    with PdfPages(output / "outage_results.pdf") as pdf:
        for name, title, subtitle, figure in figures:
            for extension in ("png", "svg", "pdf"):
                path = output / f"{name}.{extension}"
                figure.savefig(path, dpi=180, bbox_inches="tight")
                outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            pdf.savefig(figure, bbox_inches="tight")
            gallery.extend([f"## {title}", "", subtitle + ".", "", f"![{title}]({name}.png)", "", f"[SVG]({name}.svg) · [PDF]({name}.pdf)", ""])
            plt.close(figure)
    outputs["outage_results.pdf"] = hashlib.sha256((output / "outage_results.pdf").read_bytes()).hexdigest()
    (output / "README.md").write_text("\n".join(gallery))
    (evidence / "outages_plot_manifest.json").write_text(json.dumps({"bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(), "outputs_sha256": outputs}, indent=2))
    print(f"Saved {len(figures)} outage-study figures, each in PNG/SVG/PDF, and combined PDF")


if __name__ == "__main__":
    main()
