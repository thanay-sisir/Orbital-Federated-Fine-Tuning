def aggregate(
    updates,
    expected_clients,
    version,
    base_identity,
    minimum,
    expected_shapes=None,
    expected_a_hash=None,
    expected_contract=None,
):
    import torch

    accepted, rejected, seen = [], [], set()
    reference = expected_shapes
    for update in updates:
        reason = None
        client = update["client"]
        tensors = update["tensors"]
        if client not in expected_clients or client in seen:
            reason = "unexpected_or_duplicate_client"
        elif update["version"] != version or update["base_identity"] != base_identity:
            reason = "incompatible_identity_or_version"
        elif expected_a_hash is not None and update.get("fixed_A_sha256") != expected_a_hash:
            reason = "fixed_A_mismatch"
        elif expected_contract is not None and update.get("adapter_contract") != expected_contract:
            reason = "adapter_contract_mismatch"
        elif update["unique_examples"] <= 0 or not tensors:
            reason = "empty_update"
        elif any("lora_B" not in key for key in tensors):
            reason = "only_B_may_be_aggregated"
        elif any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
            reason = "nonfinite"
        else:
            shapes = {key: tuple(value.shape) for key, value in tensors.items()}
            if reference is not None and reference != shapes:
                reason = "shape_mismatch"
            else:
                reference = shapes
        if reason:
            rejected.append({"client": client, "reason": reason})
        else:
            seen.add(client)
            accepted.append(update)
    if len(accepted) < minimum:
        return None, [], rejected
    total = sum(u["unique_examples"] for u in accepted)
    result = {
        key: sum(u["tensors"][key].float() * (u["unique_examples"] / total) for u in accepted)
        for key in accepted[0]["tensors"]
    }
    return result, [u["client"] for u in accepted], rejected
