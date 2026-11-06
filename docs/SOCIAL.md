# A short four-post thread

These drafts fit the standard character budget with room for a shortened repository link. Keep each image’s experimental-scope labels visible.

## Post 1

I built Orbital Federated Fine-Tuning: five A100 clients learning from separate data shards under simulated satellite communication limits. The question: how do I keep useful training moving when links disappear?

Image: `media/01_orbital_system.png`

## Post 2

In one selected outage scenario and seed, feasibility-aware scheduling accepted 14 training steps vs 8, with about 40% less simulated wire traffic. This is a scenario-specific result, not a general speedup or flight test.

Image: `media/02_outage_results.png`

## Post 3

The tradeoff: lower conversational test loss did not improve sampled instruction following (80% base, 77% orbital). A separate three-seed study delivered more updates but had worse validation loss than fixed scheduling. Those findings matter too.

Image: `media/03_quality_tradeoffs.png`

## Post 4

Two injected coordinator crashes recovered the same final adapter in a bounded five-GPU test. Separate TCP replay verified payloads from all five clients sharing station service. Training ran on Earth; orbital constraints were simulated.

Image: `media/04_recovery_and_transfers.png`
