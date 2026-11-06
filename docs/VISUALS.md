# How to read the new diagrams

The [scheduling timeline](media/06_scheduling_timeline.png) follows client 0 in round index 2 of the selected outage experiment, using the recorded fixed and feasibility/fairness runs. Both start local training at simulated mission time 41,119.062590 seconds. Fixed scheduling performs four steps and begins uploading at 41,239.062590; it stores six of the seven chunks, ending at 41,441.700441, with an incomplete payload. Communication-aware scheduling performs two steps, starts uploading at 41,179.062590 and completes all seven chunks at 41,399.244666. At the round close of 43,200 seconds, fixed scheduling accepts no clients while communication-aware scheduling accepts clients 0 and 2. Completing a payload and accepting a round are distinct events. The figure shows verified chunk intervals, not every contact or outage. Its evidence is one client and round, not a general performance claim.

The [transfer recovery diagram](media/05_transfer_recovery.png) illustrates the saved-adapter transfer protocol. Its five boxes are schematic: they are not the measured chunk count, an elapsed-time axis or one captured contact schedule. The retained-chunk, resume and corruption-rejection behavior is supported by the [recorded TCP replay checks](orbital_sft/evidence/socket_transport_results.json) and implemented in the durable receiver. The transfer must retain a valid identity/version; verified partial chunks alone do not qualify as a full update or commit a training round. This local networking study replayed saved adapters and performed no new learning.

The [sanitized timeline extract](media/timeline_data.json) contains the plotted timestamps, chunk intervals, step counts, round participants, selected replay checks and SHA-256 hashes of their archived sources. It excludes machine-specific paths, raw training conversations and model payloads. The editable [recovery SVG](media/05_transfer_recovery.svg) and [scheduling SVG](media/06_scheduling_timeline.svg) are the source artwork; both PNG versions are 1600 × 900 pixels. To render the PNGs again with ImageMagick, run:

```bash
convert -background '#081525' docs/media/05_transfer_recovery.svg -strip docs/media/05_transfer_recovery.png
convert -background '#081525' docs/media/06_scheduling_timeline.svg -strip docs/media/06_scheduling_timeline.png
```

Different fonts or renderer versions can change image hashes. The numerical trace and its scope must remain unchanged. These additions required no new training or Slurm jobs.
