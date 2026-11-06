def effective_average(fixed_a, b_factors, counts, scaling):
    import torch

    total = sum(counts)
    if total <= 0:
        raise ValueError("Positive total weight required")
    dense = torch.zeros((b_factors[0].shape[0], fixed_a.shape[1]), dtype=torch.float64)
    for factor, count in zip(b_factors, counts, strict=True):
        dense += (factor.double() @ fixed_a.double()) * (scaling * count / total)
    return dense
