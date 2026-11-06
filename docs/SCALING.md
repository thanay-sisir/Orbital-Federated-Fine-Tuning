# What scaling would require

The completed experiments used five A100 replicas of a 27B base, restricted adapter training, one filtered conversational source and bounded mission scenarios. They do not establish the feasibility of 100-trillion-parameter pre-training, mid-training or post-training in orbit.

The transferable research question is how much useful learning reaches the shared model per unit of mission time, communication and energy. That question remains relevant as models and constellations grow. Whether this implementation scales, and where it stops being useful, requires new evidence.

## A size check, not a forecast

A dense 100-trillion-parameter model has 100,000,000,000,000 parameters. At two bytes per parameter, weights alone occupy **200 TB in decimal units**. Gradients, optimizer state, activations, checkpoints, redundancy and communication buffers add further storage. Quantization and sharding change these requirements, and each brings its own performance/quality tradeoffs. Adapter training reduces trainable state, but still needs access to the base and enough memory/compute to run it.

This arithmetic is not a hardware design or a claim that a spacecraft could host that workload. No speedup, cost, satellite count or completion-time prediction for such a model is supported by the present measurements.

## Research steps with measurable outcomes

First, identify a reference spacecraft computer and measure training time, peak memory and energy. Couple those profiles to eclipse, battery, thermal and housekeeping constraints. The present A100 measurements and assumed power ranges do not supply that calibration.

Second, extend communication experiments to changing topology, actual shared demand, inter-satellite routing, update staleness and realistic channel budgets. Compare useful accepted learning work, quality and resource use under identical mission inputs. Tightly synchronized tensor/pipeline parallel pre-training and loose federated local training have different communication requirements; their results should not be extrapolated interchangeably.

Third, test heterogeneous and changing client data, rare-client participation, checkpoint recovery at larger state sizes, corruption detection and secure update provenance. Include a ground-training alternative that accounts for raw-data downlink and initial model distribution. A space-based training approach should demonstrate an operational advantage for a specified mission rather than assume one.

Finally, repeat on unseen scenarios and representative hardware. Report failure regions and uncertainty as well as successful cases. “Mid-training” should be defined by its actual learning objective and update method; the label does not by itself determine resource requirements.

A defensible future claim would be: **“I measured where communication-aware learning remains useful under a specified spacecraft and constellation budget.”** That claim needs experiments beyond this release.
