# Shared-station payload control

Status: **passed**.

passed on 94 protocol checks; 27/27 payload/control checks passed.

The previous co-visible stress is held fixed except that outages are removed. Verified bytes below come from read-only inspection of receiver chunks, compared byte-for-byte and by SHA256 with the archived adapter.

| Client | Predicted verified chunks | Measured verified chunks | Verified payload bytes |
|---|---:|---:|---:|
| 0 | 2 | 2 | 1048576 |
| 1 | 2 | 2 | 1048576 |
| 2 | 2 | 2 | 1048576 |
| 3 | 2 | 2 | 1048576 |
| 4 | 1 | 1 | 524288 |

Total verified payload: 4718592 bytes. Jain byte-share index: 0.952941.

Clients meeting the 524288-byte requirement: 5/5.

## Failed checks

None.

## Limits

- One-host TCP replay with cloned co-visible windows; no independently propagated co-visible constellation.
- Five clients replay the same archived adapter. This experiment performs no new training or aggregation.
- The short shared-station budget tests partial-payload progress, not complete adapter delivery.
- Rate, TDMA processing guard and 360:1 mission-time compression match the preceding study; they are not calibrated flight parameters.
- The Jain index describes byte shares in this one fixed schedule; it does not establish fairness optimality.
