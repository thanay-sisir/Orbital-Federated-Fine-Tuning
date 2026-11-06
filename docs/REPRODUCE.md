# Reproduce the small experiments

Start with the model-free demos. They use Python 3.12 and the standard library, read the included numerical evidence, and write fresh results under `results/`.

```bash
PYTHONPATH=src python3.12 scripts/orbital_cpu_demo.py
PYTHONPATH=src python3.12 scripts/orbital_outages_demo.py
```

The first replays the seed-42 constrained scheduler and a small synthetic-byte durable receiver. The second replays the four selected outage conditions. These reproduce scheduling/byte-accounting behavior, not model quality or spacecraft hardware.

On a shared cluster, execute compute demonstrations and tests inside a CPU allocation. A laptop or standalone workstation can run the two small demos directly.

## Tests

The targeted protocol tests live under `tests/orbital_sft/`. They include synthetic tensors/bytes and tiny CPU training backends. GPU model weights are not needed for those tests. The full suite needs a compatible CPU-capable PyTorch installation plus the pinned configuration, serialization and test dependencies. Tests requiring excluded archived HPC artifacts explicitly skip when those files are absent; skips are not counted as passed evidence. A metadata-only prepared-data manifest is included as a seed-contract fixture, with no conversation records. Small historical configuration presets are retained where schema regression tests require them.

```bash
PYTHONPATH=src python -m pytest -c tests/orbital_sft/pytest.ini tests/orbital_sft
```

Use separate environments for training, scoring and analysis. `infra/rivanna/environment/requirements-gpu.txt` records the training stack; `requirements-analysis.lock.txt` records the plotting/coordinate environment; `requirements-ifeval.txt` records the scorer dependencies. CUDA packages do not imply that a CPU test should allocate a GPU.

## Regenerate the four result graphics

In the separate analysis environment:

```bash
python scripts/orbital_release_figures.py
```

The input is `docs/media/release_metrics.json`, derived from saved study reports. It includes source hashes and scope labels. Outputs are four 1600 × 900 PNGs plus SVG versions and an output hash manifest. Rendering versions can change metadata/output hashes; underlying numerical values must remain unchanged.

## Training reproduction

Training requires staging the exact base/tokenizer and allowed data source yourself, checking their terms, preparing the splits and completing a fresh compatibility pilot. These inputs are not part of the public repository. The original experiment used five A100-SXM4-80GB replicas and about 25.6 GiB peak allocated tensor memory per full-epoch worker; this is not a total device-memory or utilization guarantee for another environment.

The base identity is `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The dataset is `allenai/tulu-3-sft-mixture` at revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`, restricted to `ai2-adapt-dev/oasst1_converted`. See [data attribution](orbital_sft/DATA_TERMS.md).

The commands `scripts/orbital_stage.py model|data --config CONFIG` stage inputs at their configured paths. Existing destinations are immutable. The CLI `scripts/orbital_sft` supports configuration validation, data preparation, contact generation, pilots, full-epoch runs and mission runs. Run a new pilot after source/environment changes, give every experiment a fresh run identity, and update the configuration's compatibility path to that passing pilot. Archived pilot paths do not authorize changed source.

The Slurm templates are examples requiring local adaptation. Set your own account with `sbatch --account=YOUR_ACCOUNT`, select a partition and constraints available at your site, and submit from the repository root. The public templates use a placeholder account rather than someone else's allocation. GPU training remains an intentional separate action; neither CPU demo starts it.

## What the public release preserves

Numerical replay inputs and compact reports are included under `docs/orbital_sft/evidence/`. Model weights, raw conversations, generated answers, receiver databases, full run trees, machine-specific logs, caches, virtual environments and unrelated legacy code are excluded. Some scripts for full historical report collection require those separately staged artifacts; the small demos are the portable entry points.

This is a focused export of the working project, with packaging and cluster-path portability changes. Tests construct their own synthetic protocol fixtures rather than depend on private HPC files. The CPU socket demo adds an explicit `--allow-local-cpu` option for standalone workstations; GPU allocation checks are unchanged. Historical experimental hashes refer to their recorded revisions. The public tree is not claimed to be byte-identical to every historical run. Release inventory and checks are recorded separately; new GPU claims require new execution evidence.

The two additional transfer and scheduling diagrams have editable SVG sources, a sanitized trace extract and separate [rendering instructions](VISUALS.md).
