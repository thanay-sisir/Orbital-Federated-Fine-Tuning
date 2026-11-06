import numpy as np
import pytest

from swarmshield.orbital.config import Network
from swarmshield.orbital.feasibility import simulate
from swarmshield.orbital.quantized_update import decode, encode


def network():
    return Network(rate_bps=8000, chunk_bytes=100, framing_bytes=0, ack_bytes=0, acquisition_seconds=0, min_participants=2)


def test_generous_windows_deliver_equal_work():
    contacts = [{"client": i, "station": "ground", "start": 0, "end": 100, "delay_seconds": 0} for i in range(2)]
    fixed = simulate(contacts, network(), 2, 100, 100, 100, 1, policy="fixed")
    feasible = simulate(contacts, network(), 2, 100, 100, 100, 1, policy="feasible_fair")
    assert fixed["accepted_steps"] == feasible["accepted_steps"] == 8
    assert fixed["wire_bytes"] == feasible["wire_bytes"] == 400


def test_no_contact_cannot_create_a_delivered_update():
    result = simulate([], network(), 2, 100, 100, 100, 1, policy="feasible_fair")
    assert result["accepted_updates"] == result["accepted_steps"] == result["wire_bytes"] == 0


def test_admission_reduces_steps_only_when_needed_to_deliver():
    contacts = [{"client": i, "station": f"ground-{i}", "start": 0, "end": 3, "delay_seconds": 0} for i in range(2)]
    fixed = simulate(contacts, network(), 2, 3, 3, 100, 1, policy="fixed")
    feasible = simulate(contacts, network(), 2, 3, 3, 100, 1, policy="feasible_fair")
    assert fixed["accepted_updates"] == 0
    assert feasible["accepted_updates"] == 2
    assert feasible["accepted_steps"] == 4
    assert all(receipt["arrival"] <= 3 for receipt in feasible["receipts"] if receipt["complete"])


def test_quantized_delta_feedback_telescopes_and_corruption_fails():
    delta = {"layer.lora_B": np.linspace(-1, 1, 100, dtype=np.float32)}
    first, residual = encode(delta)
    reconstructed = decode(first)
    assert np.allclose(delta["layer.lora_B"] - reconstructed["layer.lora_B"], residual["layer.lora_B"])
    second, remainder = encode(delta, residual)
    combined = reconstructed["layer.lora_B"] + decode(second)["layer.lora_B"]
    assert np.allclose(2 * delta["layer.lora_B"] - combined, remainder["layer.lora_B"], atol=2e-7)
    with pytest.raises(ValueError, match="checksum"):
        decode(first[:-1] + bytes([first[-1] ^ 1]))


def test_codec_rejects_invalid_updates():
    with pytest.raises(ValueError):
        encode({"layer.lora_A": np.ones(2, dtype=np.float32)})
    with pytest.raises(ValueError):
        encode({"layer.lora_B": np.array([np.nan], dtype=np.float32)})
