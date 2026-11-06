# Training under mission-time outages

Status: passed

Missing: []

| Condition | Performed steps | Accepted steps | Updates | Failed attempts | Wire MiB | Full validation loss |
|---|---:|---:|---:|---:|---:|---:|
| no_outage_control | 64 | 16 | 4 | 0 | 160.120 | 1.359109 |
| fixed | 56 | 8 | 2 | 2 | 147.126 | 1.388090 |
| retry_backoff | 56 | 8 | 2 | 2 | 147.126 | 1.388090 |
| feasible_fair | 26 | 14 | 4 | 1 | 87.555 | 1.366927 |

Shared controls: {'same_initial_adapters': True, 'same_source': True, 'same_config_except_policy': True}

Allocated GPU hours excluding pilot: 5.3764.

All conditions share geometry, maximum work, prepared data, initialization and source. The no-outage fixed control removes exogenous failures; other conditions share one realized trace. Outages advance modeled time; verified chunks survive and failed attempts consume wire bytes. Incomplete optimizer-step work is simulated overhead, not GPU-executed partial gradients. Quality is measured on all 344 validation conversations; no held-out test tuning. One training seed and selected scenario limit generalization.
