# Integrated mission recovery

Status: passed

Missing runs: []

| Check | Passed |
|---|---|
| identical_logical_progress | True |
| identical_initial_adapter | True |
| same_config_source_environment | True |
| identical_link_reservations | True |
| one_record_per_round | True |
| two_declared_process_interruptions | True |
| saved_upload_chunks_reused | True |
| same_tensor_keys | True |
| fixed_A_exactly_equal | True |
| adapter_equal_within_tolerance | True |
| validation_loss_within_tolerance | True |

Maximum adapter absolute difference: 0.0.
Full-validation loss delta: 0.0.
Array allocated GPU hours, excluding pilot: 2.9486.

CPU uses a tiny synthetic SGD backend. GPU runs use the actual pinned language model, five replicas, two rounds and one local optimizer step per client per round. Faulted and uninterrupted runs share configuration, source and initial state. Modeled mission time is replayed; uncommitted physical computation can repeat and Slurm accounts for allocation overhead.
