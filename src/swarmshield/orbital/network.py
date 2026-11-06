import hashlib
import math
from pathlib import Path


class LinkEngine:
    def __init__(self, contacts, settings):
        self.contacts = contacts
        self.settings = settings
        self.busy = {}
        self.client_busy = {}
        self.wire_bytes = 0
        self.receipts = []
        self.identities = set()

    def transfer(self, client, size, ready, deadline, direction, identity, payload_path=None):
        if identity in self.identities:
            raise ValueError("Duplicate transfer identity")
        self.identities.add(identity)
        n = self.settings
        if size <= 0 or size > n.queue_bytes:
            raise ValueError("Transfer exceeds queue capacity or is empty")
        remaining, verified, chunks = size, 0, []
        current = ready
        while remaining:
            payload = min(remaining, n.chunk_bytes)
            wire = payload + n.framing_bytes + n.ack_bytes
            if self.wire_bytes + wire > n.byte_budget:
                break
            candidates = []
            for contact in self.contacts:
                if contact["client"] != client:
                    continue
                station = contact["station"]
                begin = max(current, contact["start"])
                duration = n.acquisition_seconds + wire * 8 / n.rate_bps
                reservations = sorted(self.busy.get(station, []) + self.client_busy.get(client, []))
                for left, right in reservations:
                    if begin < right and begin + duration > left:
                        begin = right
                finish = begin + duration
                arrival = finish + contact["delay_seconds"]
                if finish <= contact["end"] and arrival <= deadline:
                    candidates.append((arrival, station, begin, finish))
            if not candidates:
                break
            arrival, station, begin, finish = min(candidates)
            if not all(math.isfinite(v) for v in (arrival, begin, finish)):
                raise ValueError("Non-finite transfer time")
            self.busy.setdefault(station, []).append([begin, finish])
            self.client_busy.setdefault(client, []).append([begin, finish])
            self.wire_bytes += wire
            chunk_hash = None
            if payload_path is not None:
                with Path(payload_path).open("rb") as stream:
                    stream.seek(verified)
                    block = stream.read(payload)
                if len(block) != payload:
                    raise ValueError("Payload size changed during transfer")
                chunk_hash = hashlib.sha256(block).hexdigest()
            chunks.append(
                {
                    "station": station,
                    "start": begin,
                    "finish": finish,
                    "arrival": arrival,
                    "payload_bytes": payload,
                    "wire_bytes": wire,
                    "offset": verified,
                    "sha256": chunk_hash,
                }
            )
            verified += payload
            remaining -= payload
            current = arrival
        receipt = {
            "id": identity,
            "client": client,
            "direction": direction,
            "size": size,
            "verified_bytes": verified,
            "complete": remaining == 0,
            "ready": ready,
            "arrival": current if remaining == 0 else None,
            "chunks": chunks,
        }
        self.receipts.append(receipt)
        return receipt

    def snapshot(self):
        return {
            "busy": self.busy,
            "client_busy": self.client_busy,
            "wire_bytes": self.wire_bytes,
            "identities": sorted(self.identities),
        }

    def restore(self, state):
        self.busy = state["busy"]
        self.client_busy = {int(k): v for k, v in state["client_busy"].items()}
        self.wire_bytes = state["wire_bytes"]
        self.identities = set(state["identities"])
