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
    source = (ROOT / args.input).resolve()
    if not source.is_relative_to(ROOT):
        parser.error("Input must stay inside the project")
    result_path = source / "results.json"
    data = json.loads(result_path.read_text())
    destination = source / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "orbital-socket-v1"})
    artifacts = []
    with PdfPages(destination / "socket_results.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        fig.suptitle(f"Measured adapter transfer validation — {data['status']}", fontsize=15)
        colors = {"control": "#2563eb", "recovery": "#d97706", "deadline": "#64748b"}
        for condition, color in colors.items():
            rows = [r for r in data["rows"] if r["condition"] == condition and r["delivered"]]
            axes[0].scatter([r["predicted_seconds"] for r in rows], [r["measured_seconds"] for r in rows],
                            color=color, label=condition, alpha=0.8)
        maximum = max(max(r["predicted_seconds"] or 0, r["measured_seconds"] or 0) for r in data["rows"]) * 1.15
        axes[0].plot([0, maximum], [0, maximum], "--", color="#94a3b8", label="Prediction = measurement")
        axes[0].set(xlabel="Predicted completion (s)", ylabel="Measured completion (s)", xlim=(0, maximum), ylim=(0, maximum))
        axes[0].legend(fontsize=8)
        for index, condition in enumerate(colors):
            rows = [r for r in data["rows"] if r["condition"] == condition]
            axes[1].bar(index - 0.16, sum(r["predicted_request_bytes"] + r["predicted_response_bytes"] for r in rows) / 2**20,
                        width=0.3, color="#93c5fd", label="Predicted" if index == 0 else None)
            axes[1].bar(index + 0.16, sum(r["measured_request_bytes"] + r["measured_response_bytes"] for r in rows) / 2**20,
                        width=0.3, color="#2563eb", label="Measured" if index == 0 else None)
        axes[1].set(xticks=range(3), xticklabels=list(colors), ylabel="Forwarded application frames (MiB)")
        axes[1].legend()
        fig.supxlabel("One-host TCP replay · five independent client links · no training or spacecraft hardware", fontsize=9)
        for suffix in ("png", "svg", "pdf"):
            path = destination / f"01_prediction_vs_measurement.{suffix}"
            fig.savefig(path, dpi=170)
            artifacts.append(path)
        pdf.savefig(fig)
        plt.close(fig)
    artifacts.append(destination / "socket_results.pdf")
    manifest = {"input_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "files": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(destination))


if __name__ == "__main__":
    main()
