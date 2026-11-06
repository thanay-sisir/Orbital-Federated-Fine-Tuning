import pytest

torch = pytest.importorskip("torch")

from evaluator.orbital_sft.aggregation_reference import effective_average
from swarmshield.orbital.aggregation import aggregate


def update(client, tensor, count=1, version=0):
    return {
        "client": client,
        "version": version,
        "base_identity": "fixture@commit",
        "unique_examples": count,
        "tensors": {"layer.lora_B.default.weight": tensor},
    }


def test_fixed_a_matches_dense_effective_weight_oracle():
    a = torch.tensor([[1.0, 2.0, -1.0], [0.0, 3.0, 4.0]])
    b0, b1 = torch.tensor([[2.0, 1.0], [1.0, -1.0]]), torch.tensor([[0.0, 3.0], [2.0, 1.0]])
    result, members, _ = aggregate(
        [update(0, b0, 2), update(1, b1, 3)], {0, 1}, 0, "fixture@commit", 2
    )
    expected = effective_average(a, [b0, b1], [2, 3], 2)
    assert members == [0, 1]
    torch.testing.assert_close((result["layer.lora_B.default.weight"] @ a).double() * 2, expected)


def test_duplicate_stale_and_nonfinite_rejected():
    good = update(0, torch.ones(2, 2))
    stale = update(1, torch.ones(2, 2), version=1)
    bad = update(2, torch.full((2, 2), float("nan")))
    result, members, rejected = aggregate(
        [good, good, stale, bad], {0, 1, 2}, 0, "fixture@commit", 2
    )
    assert result is None and members == [] and len(rejected) == 3


def test_two_factor_average_is_not_dense_average():
    a0, a1 = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    b0, b1 = torch.tensor([[1.0], [0.0]]), torch.tensor([[0.0], [1.0]])
    assert not torch.equal((b0 + b1) / 2 @ ((a0 + a1) / 2), (b0 @ a0 + b1 @ a1) / 2)


def test_global_contract_is_authoritative():
    tensors = torch.ones(2, 2)
    forged = update(0, tensors)
    forged["fixed_A_sha256"] = "wrong"
    result, _, rejected = aggregate(
        [forged], {0}, 0, "fixture@commit", 1, expected_a_hash="expected"
    )
    assert result is None and rejected[0]["reason"] == "fixed_A_mismatch"


def test_shapes_must_match_ground_state():
    result, _, rejected = aggregate(
        [update(0, torch.ones(3, 2))],
        {0},
        0,
        "fixture@commit",
        1,
        expected_shapes={"layer.lora_B.default.weight": (2, 2)},
    )
    assert result is None and rejected[0]["reason"] == "shape_mismatch"
