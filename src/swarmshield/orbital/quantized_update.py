import hashlib
import json
import math
import struct

import numpy as np


MAGIC = b"ORBQ8V1\0"


def encode(delta, residual=None):
    residual = residual or {}
    if not delta or set(residual) - set(delta):
        raise ValueError("Empty delta or unexpected residual")
    blocks, metadata, remaining, offset = [], [], {}, 0
    for name in sorted(delta):
        if "lora_B" not in name:
            raise ValueError("Only B updates may be encoded")
        value = np.asarray(delta[name], dtype=np.float32)
        previous = np.asarray(residual.get(name, np.zeros_like(value)), dtype=np.float32)
        if value.size == 0 or value.shape != previous.shape or not np.isfinite(value).all() or not np.isfinite(previous).all():
            raise ValueError("Invalid tensor or residual")
        corrected = value + previous
        if not np.isfinite(corrected).all():
            raise ValueError("Nonfinite corrected update")
        maximum = float(np.max(np.abs(corrected)))
        scale = maximum / 127 if maximum else 1.0
        codes = np.clip(np.rint(corrected / scale), -127, 127).astype(np.int8)
        data = codes.tobytes()
        blocks.append(data)
        metadata.append({"name": name, "shape": list(value.shape), "offset": offset, "count": value.size, "scale": scale, "sha256": hashlib.sha256(data).hexdigest()})
        remaining[name] = corrected - codes.astype(np.float32) * scale
        offset += len(data)
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    return MAGIC + struct.pack("<Q", len(header)) + header + b"".join(blocks), remaining


def decode(payload):
    if payload[:8] != MAGIC or len(payload) < 16:
        raise ValueError("Invalid update header")
    size = struct.unpack("<Q", payload[8:16])[0]
    if size > len(payload) - 16:
        raise ValueError("Truncated update header")
    metadata = json.loads(payload[16:16 + size])
    data, result, consumed = payload[16 + size:], {}, 0
    for item in metadata:
        name, shape, count, scale = item["name"], item["shape"], item["count"], item["scale"]
        if "lora_B" not in name or name in result or item["offset"] != consumed:
            raise ValueError("Invalid update name or layout")
        if any(not isinstance(v, int) or v < 1 for v in shape) or math.prod(shape) != count or count < 1 or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid update shape or scale")
        block = data[consumed:consumed + count]
        if len(block) != count or hashlib.sha256(block).hexdigest() != item["sha256"]:
            raise ValueError("Update checksum mismatch")
        result[name] = np.frombuffer(block, dtype=np.int8).astype(np.float32).reshape(shape) * scale
        consumed += count
    if consumed != len(data) or not result:
        raise ValueError("Extra or empty update data")
    return result
