import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-originals", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "docs/orbital_sft"
    manifest = json.loads((directory / "evidence/manifest.json").read_text())
    bundle = directory / "evidence/plot_data.json"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest["plot_bundle_sha256"]:
        raise AssertionError("Plot bundle checksum mismatch")
    for name, expected in manifest["outputs_sha256"].items():
        if hashlib.sha256((directory / "figures" / name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Figure checksum mismatch: {name}")
    missing = []
    for name, expected in manifest["source_sha256"].items():
        path = ROOT / name
        if not path.exists():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Original evidence checksum mismatch: {name}")
    if args.require_originals and missing:
        raise AssertionError(f"Missing {len(missing)} original artifacts")
    data = json.loads(bundle.read_text())
    assert len(data["runs"]) == 9 and data["repeat"]["incomplete"] == []
    assert data["geometry"]["status"] == data["recovery"]["status"] == "passed"
    assert data["protocol_checks"]["failures"] == data["protocol_checks"]["errors"] == 0
    print(json.dumps({"status": "passed", "verified_figure_files": len(manifest["outputs_sha256"]),
                      "originals_unavailable": len(missing), "protocol_checks": data["protocol_checks"]}, indent=2))


if __name__ == "__main__":
    main()
