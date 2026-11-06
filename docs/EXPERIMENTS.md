# What was measured

This project asks how to keep distributed training productive when communication is limited and constantly changing. The experiments separate three kinds of evidence: real five-GPU training with modeled communication, CPU simulation, and actual local TCP replay of saved adapters. They are not presented as one flight deployment or one end-to-end socket-based GPU training run.

## Training recipe

Five A100-SXM4-80GB GPUs each held a complete replica of the pinned `Qwen/Qwen3.8-27B` base. The NF4/BF16 adapter recipe used rank 8, alpha 16, selected language q/v projections and fixed-A/trainable-B factors. Eligible B updates were averaged using unique-example weights. The base and fixed factors were pre-staged.

The reviewed OpenAssistant source from `allenai/tulu-3-sft-mixture` produced 6,925 accepted conversations: 6,216 train, 344 validation and 365 test. Training shards contained 1,244 / 1,244 / 1,244 / 1,244 / 1,240 examples. Assistant responses supplied supervised targets; overlength conversations beyond 1,024 tokens were dropped. Two full epochs were completed. The selected epoch-two adapter was chosen using full validation loss.

This is one filtered conversational source, not the entire Tulu mixture or strong evidence of heterogeneous client data. Local exact deduplication does not establish absence of benchmark or pretraining contamination.

## Quality and the connected control

| Model | Validation answer loss | Test answer loss | IFEval strict | IFEval loose |
|---|---:|---:|---:|---:|
| Base | 1.440864 | 1.534522 | 80% | 84% |
| Orbital epoch two | 1.036431 | 1.117775 | 77% | 80% |
| Connected epoch two | 1.036380 | 1.117747 | 76% | 79% |

Validation uses all 344 conversations; test loss uses all 365. The IFEval results use a fixed 100-prompt sample from 541 prompts, official pinned checkers, greedy generation and a 2,048-token output cap. This is not a full-benchmark leaderboard submission. The orbital strict-compliance change versus base was −3 percentage points, with a paired bootstrap interval of [−12, +6] points. No broad instruction-following improvement is established.

The connected comparison is federated, not centralized single-GPU SFT. Historical training revisions and topology differences are documented in the original studies. Epoch-one test loss had already been exposed during development; epoch selection used full validation loss and was frozen before the final comparison. The complete test-development history is therefore not described as untouched. Under generous deadlines all five clients contributed, so similar final quality is expected. Lower conversational answer loss is a useful result, but it is not the same as better general accuracy.

## Repeated constrained-mission training

Three policies ran at three training seeds, giving nine actual GPU runs under one fixed constrained geometry. Preparation seed 42 was held fixed; initial adapters matched across policies within each seed and differed across training seeds.

| Policy | Accepted / performed steps | Accepted updates | Simulated wire MiB | Validation loss, mean ± sample SD |
|---|---:|---:|---:|---:|
| Connected | 120 / 120 | 30 | 390.323 | 1.253132 ± 0.004171 |
| Fixed | 64 / 72 | 16 | 221.183 | 1.246644 ± 0.004531 |
| Feasibility/fairness | 68 / 68 | 18 | 234.194 | 1.252701 ± 0.004417 |

Feasibility/fairness delivered 12.5% more accepted updates and 6.25% more accepted optimizer steps than fixed, using 5.88% more modeled traffic. Fixed had lower validation loss on every seed. Three seeds in one geometry provide a limited repeatability check, not a universal scheduler ranking.

## Unexpected mission-time outages

The CPU study covered 648 policy trajectories over six geometries, two radio rates, two assumed compute profiles, three failure profiles and three fault seeds. No-outage fault seeds repeat deterministic cases. They are not independent repetitions. A baseline-only rule selected one pressure case before GPU quality results were available.

| Five-GPU condition | Performed steps | Accepted steps | Accepted updates | Simulated wire MiB | Validation loss |
|---|---:|---:|---:|---:|---:|
| No-outage fixed control | 64 | 16 | 4 | 160.120 | 1.359109 |
| Fixed with outages | 56 | 8 | 2 | 147.126 | 1.388090 |
| Retry/backoff with outages | 56 | 8 | 2 | 147.126 | 1.388090 |
| Feasibility/fairness with outages | 26 | 14 | 4 | 87.555 | 1.366927 |

In this selected scenario and training seed, feasibility/fairness accepted 75% more work than fixed with approximately 40.5% less modeled wire traffic. It also had lower validation loss than fixed, while remaining worse than the no-outage control. These percentages are not throughput speedups, RF measurements or general model-quality gains.

The policy receives nominal contacts and current observations, not future realized failures. Failed prefixes consume modeled bytes/time. Completed verified chunks persist. Client-power interruptions restart incomplete simulated steps; the physical GPUs execute the completed steps, not the discarded partial gradients. This distinction matters when interpreting simulated compute cost.

## Recovery

The coordinator binds cached updates to their parent/configuration/cursor/work contract. A transactional ledger commits adapter bytes and round progress together; retries do not duplicate committed rounds. A five-GPU recovery control and interrupted run shared inputs and source. After two deliberate process exits, final adapter maximum absolute difference and full-validation loss difference were both 0.0. Eleven comparison checks passed.

This was a bounded two-round experiment with one optimizer step per client per round, under connected contacts. It validates the tested process-recovery mechanism, not hardware power-loss recovery, radiation resilience, zero replay overhead or arbitrary mid-step optimizer checkpointing.

## Real network replay

Five independent client processes transferred saved adapters through a protocol-aware TCP proxy to a separate coordinator on one CPU host. Initial control, interruption/corruption and unavailable-client deadline conditions passed 121 checks. These are actual network transfers, but they replay saved adapters and do not train through the socket path.

Ten later replay cases constrained backend sockets to orbit-derived TDMA slots. All 3,069 case checks passed; the maximum confirmed-client timing residual was 0.0696 wall seconds. A 24-hour simulated mission was uniformly compressed to at most 240 wall seconds. The 64 Mbps shaped rate, 250 ms slots and 150 ms processing guard are laboratory assumptions. The new protocol differs from the older numerical LinkEngine.

The original orbital schedules had no overlapping shared-station passes. A separately labeled co-visible stress case cloned contact windows across clients. Its outages blocked payload attempts, so a no-outage control was added without changing nominal settings. That follow-up passed 28 tests, 94 protocol checks and 27 payload/control checks. All five clients stored verified chunks: nine chunks totaling 4.5 MiB. Actual byte/hash checks, prediction checks and exclusive station service passed. This proves partial payload progress, not complete adapter delivery in that short shared budget.

## Geometry, compression and cost

An expanded coordinate/event cross-check reproduced 245 synthetic contact intervals across six geometries, with maximum boundary discrepancy 0.2142 seconds. Both pipelines share SGP4; this is not independent physical validation of an orbit or radio link.

A separate CPU int8-delta codec reduced ten archived client/epoch payloads from approximately 6.82 MB to 1.71 MB, about 3.98× smaller. Compressed GPU learning quality remains untested.

Recorded allocation scopes are 12.4333 GPU hours for the nine-run repeated study, 5.3764 GPU hours for the four-condition outage array, and 2.9486 GPU hours for the recovery array; each excludes its pilot. The ten contact replay tasks used 1.25 allocated CPU hours excluding tests/report. These are scoped allocation costs, not a full-project bill, GPU-utilization measurement or dollar estimate.

## What the evidence does not establish

There is no flight deployment, calibrated onboard compute/energy/thermal budget, physical RF link, inter-satellite network, authenticated adversarial-update protocol, 100-trillion-parameter experiment or general capability win. Checksums assume trusted manifests. Storage/process recovery tests do not emulate broken hardware or filesystem power loss. The code and measurements form a experimental framework with explicit limits.

Numerical source data for the four public graphics is in [media/release_metrics.json](media/release_metrics.json). The release includes compact reports and replay inputs under `orbital_sft/evidence/`. Large run directories, model weights, datasets, generated answers and machine-specific logs are excluded.
