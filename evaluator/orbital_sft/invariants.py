from collections import defaultdict
from itertools import pairwise


def check_receipts(receipts, rate_bps, acquisition_seconds):
    occupied = defaultdict(list)
    total = 0
    for receipt in receipts:
        payload = 0
        for chunk in receipt["chunks"]:
            if chunk["start"] < receipt["ready"]:
                raise AssertionError("Transfer before payload ready")
            airtime = chunk["finish"] - chunk["start"] - acquisition_seconds
            if chunk["wire_bytes"] * 8 > airtime * rate_bps + 1e-5:
                raise AssertionError("Capacity exceeded")
            if chunk["offset"] != payload:
                raise AssertionError("Unverified chunk gap")
            payload += chunk["payload_bytes"]
            total += chunk["wire_bytes"]
            for resource in (f"station:{chunk['station']}", f"client:{receipt['client']}"):
                occupied[resource].append((chunk["start"], chunk["finish"]))
        if payload != receipt["verified_bytes"]:
            raise AssertionError("Byte accounting mismatch")
        if receipt["complete"] and payload != receipt["size"]:
            raise AssertionError("Incomplete payload accepted")
    for intervals in occupied.values():
        intervals.sort()
        for left, right in pairwise(intervals):
            if left[1] > right[0] + 1e-9:
                raise AssertionError("Half-duplex resource overlap")
    return total
