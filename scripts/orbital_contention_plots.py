import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    directory = (ROOT / args.input).resolve()
    if not directory.is_relative_to(ROOT):
        parser.error("Input must stay inside the project")
    path = directory / "payload_validation.json"
    result = json.loads(path.read_text())
    validation = result["payload_validation"]
    execution = directory / "execution"
    proxy = json.loads((execution / "proxy.json").read_text())
    manifest = json.loads((execution / "manifest.json").read_text())
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "contention-payload-control-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    rows = validation["clients"]
    clients = [r["client"] for r in rows]
    axes[0].bar([c-.16 for c in clients], [r["predicted_verified_payload_bytes"] / 2**20 for r in rows],
                width=.3, label="Predicted", color="#93c5fd")
    axes[0].bar([c+.16 for c in clients], [r["measured_verified_payload_bytes"] / 2**20 for r in rows],
                width=.3, label="Measured, hash-verified", color="#2563eb")
    axes[0].axhline(validation["minimum_bytes_per_client"] / 2**20, color="#d97706", linestyle="--", label="Required per client")
    axes[0].set(xticks=clients, xlabel="Client", ylabel="Verified partial adapter payload (MiB)", title="Payload progress excludes status messages")
    axes[0].legend(fontsize=8)
    payload_events = [e for e in proxy["events"] if e.get("op") == "put" and e.get("outcome") == "complete"]
    if payload_events:
        slot = manifest["slots"][payload_events[0]["slot"]]
        contact = manifest["contacts"][slot["contact"]]
        start, end = contact["start"], contact["end"]
        colors = ["#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626"]
        for c in clients:
            axes[1].broken_barh([(start, end-start)], (c-.35, .7), facecolor="#e2e8f0")
        for s in manifest["slots"]:
            if start <= s["start"] < end:
                axes[1].broken_barh([(s["start"], s["end"]-s["start"])], (s["client"]-.28,.56),
                                    facecolor=colors[s["client"]], alpha=.35)
        for event in proxy["events"]:
            if start <= event.get("link_open", -1) < end and "link_close" in event:
                axes[1].broken_barh([(event["link_open"],event["link_close"]-event["link_open"])],
                                    (event["client"]-.12,.24), facecolor=colors[event["client"]])
        axes[1].set(xlim=(start-.02,end+.02), yticks=clients, xlabel="Compressed wall-clock seconds", ylabel="Client",
                    title="Shared station: allocations and actual socket service")
    else:
        axes[1].axis("off")
        axes[1].text(.05,.5,"No completed payload transactions — validation failed", transform=axes[1].transAxes)
    fig.suptitle(f"No-outage shared-station control — {result['status']}", fontsize=15)
    fig.supxlabel("Saved-adapter replay on one CPU node · partial payloads, not complete model updates · no GPUs or training", fontsize=9)
    artifacts = []
    for suffix in ("png", "svg", "pdf"):
        target = figures / f"payload_contention.{suffix}"
        fig.savefig(target, dpi=160, bbox_inches="tight")
        artifacts.append(target)
    target = figures / "contention_results.pdf"
    with PdfPages(target) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    artifacts.append(target)
    plt.close(fig)
    (figures / "manifest.json").write_text(json.dumps({"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "files": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}, indent=2) + "\n")
    print(str(figures))


if __name__ == "__main__":
    main()
