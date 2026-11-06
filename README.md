# Orbital Federated Fine-Tuning

**Keeping distributed training productive when communication is limited and constantly changing.**

Five GPU clients fine-tune a shared model while a simulation limits when their updates can reach the coordinator. The project measures useful accepted training work, model quality, communication cost and recovery after failures.

![System overview: five clients, orbital contact constraints and a shared coordinator](docs/media/01_orbital_system.png)

Training ran on Earth. Satellite contacts and spacecraft operating constraints were simulated. Later networking experiments replayed saved adapters over actual local TCP connections; they did not perform new training through that socket path.

## Situation

A client can finish training and still miss its chance to upload an update. Limited bandwidth, changing visibility and unexpected outages make completed computation different from useful learning delivered to the shared model.

## Task

Build a reproducible five-client fine-tuning prototype and determine whether communication-aware scheduling improves accepted work under a fixed mission budget. Compare quality and cost against connected and fixed-scheduling controls, then test delivery and recovery correctness.

## Why start here?

Five clients are enough to exercise separate data shards, competing communication needs and a shared aggregation point, while keeping individual updates easy to trace. It is a manageable first experiment, not evidence that the same design scales to hundreds of spacecraft.

Adapter fine-tuning keeps the amount of trainable and transmitted state small enough for repeatable experiments. Keeping one adapter factor fixed gives clients a common update representation and makes the weighted aggregation well defined. The 27B base supplies a real training workload on the available GPUs; it is not a claim that this model fits an onboard computer.

One reviewed conversational source keeps the first quality comparison controlled. A connected federated baseline helps reveal whether a difference comes from communication restrictions. Repeated seeds check whether a scheduling result survives a change in training initialization. The later outage and socket tests ask narrower questions about failure handling and delivery correctness. Each stage adds evidence without treating one passing experiment as proof of the whole space environment.

## Action

- Ran real adapter SFT on **five A100 GPUs**, with complete replicas of a pinned **Qwen3.8-27B** base and five disjoint data shards.
- Completed two full epochs over **6,216 training conversations per epoch**. Used separate **344-example validation** and **365-example test** splits from one reviewed OpenAssistant source in Tulu.
- Modeled orbital visibility, shared radio service, bandwidth, deadlines and unexpected outages. Scheduling used nominal contacts and observations rather than future failure information.
- Compared three schedulers across **nine GPU runs and three training seeds**. Audited **648 CPU outage trajectories** across six geometries.
- Added durable verified chunks, version checks and transactional round recovery. Tested actual TCP adapter replay under contact schedules and shared-station limits.

## Result

### Useful work under one outage scenario

| Measure | Fixed scheduling | Feasibility / fairness |
|---|---:|---:|
| Accepted optimizer steps | 8 | **14** |
| Simulated wire traffic | 147.126 MiB | **87.555 MiB** |
| Full-validation answer loss | 1.388090 | **1.366927** |

That is **75% more accepted work with approximately 40.5% less modeled traffic** in one selected scenario and training seed. It is not a general accuracy gain or an actual network-throughput speedup. The no-outage control still had lower validation loss, at 1.359109.

![Accepted training steps and modeled communication in the selected outage experiment](docs/media/02_outage_results.png)

### The tradeoffs matter

In the separate three-seed study, feasibility/fairness delivered **12.5% more accepted updates** than fixed scheduling, with **5.88% more modeled traffic**, but fixed had better validation loss on every seed.

The selected full-epoch orbital adapter reduced held-out answer loss from **1.5345 to 1.1178**. A connected federated control reached **1.1177**, almost the same result under generous deadlines. Strict IFEval compliance on a fixed 100-prompt sample was **80% base, 77% orbital and 76% connected**. Lower answer loss did not establish a broader instruction-following improvement.

![Held-out conversational loss and sampled instruction-following results](docs/media/03_quality_tradeoffs.png)

### Reliability and actual transfers

- After **two deliberate coordinator process interruptions**, a bounded five-GPU recovery run matched the uninterrupted final adapter and validation loss with observed differences of **0.0**.
- **Ten contact-aware TCP replay cases** passed their checks. The largest confirmed-client timing residual was **0.0696 seconds** against the new slotted protocol predictions.
- A no-outage shared-station control transferred **nine independently verified adapter chunks totaling 4.5 MiB across all five clients**. Predicted stored positions/bytes and actual exclusive station service matched. This is partial payload progress, not complete new model updates.

![Separate GPU recovery and CPU shared-station payload-control results](docs/media/04_recovery_and_transfers.png)

## Try the CPU replay

The two small demonstrations use Python 3.12 and the standard library. They reproduce scheduling counts and receiver behavior from the included numerical evidence, without weights, raw conversations, GPUs or network downloads.

```bash
PYTHONPATH=src python3.12 scripts/orbital_cpu_demo.py
PYTHONPATH=src python3.12 scripts/orbital_outages_demo.py
```

On a shared cluster, run compute demonstrations inside a CPU allocation. See [reproduction instructions](docs/REPRODUCE.md) for tests, environments and intentional training reproduction.

## What we learned

Communication scheduling can change how much completed training actually reaches the shared model. In the selected outage case, choosing work that could fit the available contacts improved accepted work and reduced modeled traffic. Under generous deadlines, orbital and connected training produced almost the same test loss. Across three repeated seeds, more accepted updates still did not give the proposed scheduler better validation loss.

Those results point to a useful engineering lesson: measure delivery, learning quality and resource cost together. The recovery and transfer tests strengthen confidence in the implemented mechanisms, while the remaining hardware and mission assumptions define where that confidence stops.

Public release check: **95 tests passed, 4 archived-artifact cases explicitly skipped**, and both portable CPU replays passed. [Validation details](docs/RELEASE_VALIDATION.md).

## Scope

This is a ground-based research prototype. It does not establish flight readiness, measured RF behavior, calibrated onboard power/thermal feasibility or 100-trillion-parameter training. The connected control is federated, not centralized single-GPU SFT. Some experiments use one scenario or a small number of seeds. The socket and GPU studies are separate evidence, not one live socket-based GPU training deployment.

The ideas are relevant to larger systems, but scaling them to pre-training, mid-training or post-training at 100T parameters is a research question. [The scaling note](docs/SCALING.md) explains what would need to be measured before making such a claim.

## Explore

- [Experimental methods, results and limitations](docs/EXPERIMENTS.md)
- [Reproduce the demos and inspect evidence](docs/REPRODUCE.md)
- [Future scaling: questions rather than forecasts](docs/SCALING.md)
- [Numerical inputs behind the four graphics](docs/media/release_metrics.json)
- [Short description and shareable images](docs/SOCIAL.md)

The public release contains focused source, tests, configurations, numerical evidence and figures. Model weights, datasets, environments, credentials, raw run logs and unrelated legacy application files are excluded. Model/data terms and attribution are documented separately from the project’s experimental claims.
