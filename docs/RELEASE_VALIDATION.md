# Public release validation

The focused public export was checked on a CPU Slurm allocation using Python 3.12.12 and the recorded PyTorch 2.9 environment. No GPU training was launched for release preparation.

- **95 tests passed.**
- **4 archived-artifact integration cases were explicitly skipped**, because their original large HPC run files are excluded. They are not counted as passed.
- Both included standard-library CPU replay demonstrations completed successfully and reproduced their recorded scheduling and byte-accounting checks.
- Python, JSON and shell syntax checks passed.
- All public Markdown links and the four PNG/four SVG graphic hashes were checked.
- The export was screened for private cluster paths, credential patterns and disallowed data/model/cache/log files before staging.

The export retains the small independent orbital reference checker and metadata-only schema/seed fixtures required by the tests. Early export checks caught missing fixture dependencies; those were fixed before the passing run. No historical experimental result was changed to make the release pass.

Packaging changes include a focused Python project definition, portable cluster-path handling, self-contained synthetic CPU fixtures and explicit standalone CPU opt-in for the socket demo. Historical scientific measurements refer to their archived source revisions. Reproducing GPU experiments on this release requires new staging and compatibility gates.

The graphics use recorded quantities and include scenario limits, negative findings and the separation between real computation and simulated spacecraft conditions. They do not claim flight deployment, agency endorsement or demonstrated 100T-scale training.
