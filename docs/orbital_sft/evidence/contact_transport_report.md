# Orbital contact socket replay

Status: **passed**

CPU array: 22525257; tests: 22525191. Allocated CPU hours excluding tests/report: 1.2500.

| Case | Predicted eligible | Measured eligible | Confirmed clients | Forwarded MiB | Outage slots | Checks |
|---|---:|---:|---:|---:|---:|---|
| 450km-1site-control | 0 | 0 | 0 | 29.507 | 0 | passed |
| 450km-1site-seed101 | 0 | 0 | 0 | 19.006 | 15 | passed |
| 450km-1site-seed102 | 0 | 0 | 0 | 17.506 | 17 | passed |
| 550km-2site-control | 5 | 5 | 5 | 32.534 | 0 | passed |
| 550km-2site-seed101 | 5 | 5 | 5 | 32.535 | 16 | passed |
| 550km-2site-seed102 | 5 | 5 | 5 | 32.535 | 14 | passed |
| 700km-1site-control | 5 | 5 | 5 | 32.534 | 0 | passed |
| 700km-1site-seed101 | 5 | 5 | 4 | 32.535 | 19 | passed |
| 700km-1site-seed102 | 5 | 5 | 5 | 32.535 | 14 | passed |
| co-visible-stress | 0 | 0 | 0 | 0.001 | 10 | passed |

## Failed or missing checks

None.

Missing cases: [].

## Limits

- Actual TCP bytes are replayed on one CPU node; no new training, aggregation, flight hardware or onboard energy calibration.
- Three archived synthetic SGP4 geometries are uniformly compressed by 360:1: a 24-hour mission is replayed in at most 240 seconds.
- The shaped wall-clock rate is 64 Mbps, equivalent to 177777.78 bits per modeled mission second; this is a declared sensitivity, not a flight-radio measurement.
- Fixed 250 ms TDMA slots include a 150 ms processing guard. Station/client reservations are exclusive; unused allocations are not reassigned.
- Outages are synthetic slot-aligned failures, with two selected slot starts per station and client, each lasting one to three slots. Sub-slot outage behavior is not tested here.
- Original orbital contacts have no shared-station overlap. The tenth case deliberately clones one client's contact times across five clients to stress contention; it is not an independently propagated constellation.
- The proxy ingress is local and unshaped, buffering one bounded request per client. Actual proxy-to-coordinator sockets are opened only during contact slots and closed after each transaction.
- Byte accounting covers shaped application frames, excluding TCP/IP overhead and inbound buffering. Measured timing includes CPU/filesystem overhead.
- Predictions model the new slotted socket protocol, not the older LinkEngine. Shared contact inputs and code audits establish implementation consistency, not independent physical validation.
- Eligibility is durable acceptance of a saved adapter, separate from client confirmation and from a training aggregation. Acknowledgment loss can leave an eligible update unconfirmed.
