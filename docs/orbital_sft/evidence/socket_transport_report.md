# Measured socket-transfer validation

Status: **passed**; 121/121 checks passed.

Measured loopback TCP application-frame transport, five client processes, one coordinator and one protocol-aware proxy. Saved adapter replay; no model loading, training or aggregation.

Pre-execution protocol serialization/latency/outage model, not a replay of the orbital LinkEngine. Poll phase approximated by half a retry interval. CPU, filesystem and kernel overhead are measured residuals.

| Condition | Client | Delivered | Predicted s | Measured s | Timing error s | Forwarded request bytes |
|---|---:|---|---:|---:|---:|---:|
| control | 0 | True | 7.425 | 8.438 | 1.013 | 6822996 |
| control | 1 | True | 7.425 | 8.762 | 1.337 | 6822996 |
| control | 2 | True | 7.425 | 8.702 | 1.277 | 6822996 |
| control | 3 | True | 7.425 | 8.490 | 1.066 | 6822996 |
| control | 4 | True | 7.425 | 8.542 | 1.118 | 6822996 |
| recovery | 0 | True | 9.636 | 10.578 | 0.942 | 6954207 |
| recovery | 1 | True | 9.485 | 10.368 | 0.883 | 6823058 |
| recovery | 2 | True | 7.827 | 8.697 | 0.870 | 7085279 |
| recovery | 3 | True | 7.425 | 8.274 | 0.849 | 6822996 |
| recovery | 4 | True | 7.425 | 8.210 | 0.786 | 6822996 |
| deadline | 0 | True | 7.425 | 8.589 | 1.165 | 6822996 |
| deadline | 1 | True | 7.425 | 8.549 | 1.124 | 6822996 |
| deadline | 2 | True | 7.425 | 8.473 | 1.049 | 6822996 |
| deadline | 3 | True | 7.425 | 8.296 | 0.871 | 6822996 |
| deadline | 4 | False | — | — | — | 0 |

## Failed checks

None.

## Limits

- One host, five independent per-client rate limits; no shared station contention, orbital geometry or flight hardware.
- Byte counts cover application frames forwarded by the proxy, excluding TCP/IP overhead, dropped ACK bytes on the backend leg and bytes buffered on the ingress leg.
- Proxy buffers at most one bounded chunk per handler; this is a protocol-aware emulator, not packet-level RF emulation.
- Faults trigger at declared chunk boundaries; client 4 is unavailable for the entire deadline case.
- Same archived adapter is replayed by all five clients; these are not five new learned updates.
- Trusted manifest checksums detect corruption; no sender authentication or cryptographic transport security.
