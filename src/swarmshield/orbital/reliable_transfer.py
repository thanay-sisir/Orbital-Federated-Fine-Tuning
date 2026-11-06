import hashlib
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TransferSpec:
    transfer_id: str
    client: int
    version: int
    base_identity: str
    size: int
    chunk_bytes: int
    payload_sha256: str
    chunk_sha256: tuple[str, ...]

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", self.transfer_id):
            raise ValueError("Invalid transfer identity")
        for value in (self.client, self.version, self.size, self.chunk_bytes):
            if type(value) is not int:
                raise ValueError("Integer transfer metadata required")
        if self.client < 0 or self.version < 0 or self.size <= 0:
            raise ValueError("Invalid client, version or size")
        if not 1 <= self.chunk_bytes <= 4 * 1024 * 1024 or not self.base_identity:
            raise ValueError("Invalid chunk size or base identity")
        count = (self.size + self.chunk_bytes - 1) // self.chunk_bytes
        if len(self.chunk_sha256) != count:
            raise ValueError("Chunk manifest length mismatch")
        if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in (self.payload_sha256, *self.chunk_sha256)):
            raise ValueError("Invalid SHA256 manifest")

    def encoded(self):
        self.validate()
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_bytes(cls, transfer_id, client, version, base_identity, data, chunk_bytes=1048576):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 4 * 1024 * 1024:
            raise ValueError("Invalid chunk size")
        result = cls(transfer_id, client, version, base_identity, len(data), chunk_bytes,
                     sha256(data), tuple(sha256(data[i:i + chunk_bytes]) for i in range(0, len(data), chunk_bytes)))
        result.validate()
        return result


class DurableReceiver:
    def __init__(self, path, base_identity, queue_bytes=1073741824):
        if type(queue_bytes) is not int or queue_bytes <= 0 or not base_identity:
            raise ValueError("Invalid receiver settings")
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), base TEXT NOT NULL, version INTEGER NOT NULL, queue_bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS transfers (id TEXT PRIMARY KEY, spec TEXT NOT NULL, size INTEGER NOT NULL, version INTEGER NOT NULL, client INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (transfer_id TEXT NOT NULL REFERENCES transfers(id), position INTEGER NOT NULL, data BLOB NOT NULL, PRIMARY KEY(transfer_id, position));
            CREATE TABLE IF NOT EXISTS eligible (transfer_id TEXT PRIMARY KEY REFERENCES transfers(id), version INTEGER NOT NULL, client INTEGER NOT NULL, payload_sha256 TEXT NOT NULL, UNIQUE(version, client));
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO settings VALUES (1, ?, 0, ?)", (base_identity, queue_bytes))
        settings = self.db.execute("SELECT * FROM settings").fetchone()
        if settings["base"] != base_identity or settings["queue_bytes"] != queue_bytes:
            self.close()
            raise ValueError("Receiver identity or capacity differs from persisted settings")

    def close(self):
        self.db.close()

    def _begin(self):
        self.db.execute("BEGIN IMMEDIATE")

    def advance(self, version):
        if type(version) is not int:
            raise ValueError("Integer version required")
        with self.db:
            self._begin()
            current = self.db.execute("SELECT version FROM settings").fetchone()[0]
            if version < current:
                raise ValueError("Version rollback rejected")
            self.db.execute("UPDATE settings SET version=?", (version,))
            self.db.execute("DELETE FROM chunks WHERE transfer_id IN (SELECT id FROM transfers WHERE version < ? AND id NOT IN (SELECT transfer_id FROM eligible))", (version,))

    def _live(self, spec):
        settings = self.db.execute("SELECT * FROM settings").fetchone()
        if spec.base_identity != settings["base"] or spec.version != settings["version"]:
            raise ValueError("Stale version or incompatible base identity")

    def begin(self, spec):
        encoded = spec.encoded()
        with self.db:
            self._begin()
            self._live(spec)
            existing = self.db.execute("SELECT spec FROM transfers WHERE id=?", (spec.transfer_id,)).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise ValueError("Transfer identity reused with different metadata")
                return
            settings = self.db.execute("SELECT * FROM settings").fetchone()
            reserved = self.db.execute("SELECT COALESCE(SUM(size),0) FROM transfers WHERE version=? AND id NOT IN (SELECT transfer_id FROM eligible)", (spec.version,)).fetchone()[0]
            if reserved + spec.size > settings["queue_bytes"]:
                raise ValueError("Receiver queue capacity exceeded")
            self.db.execute("INSERT INTO transfers VALUES (?, ?, ?, ?, ?)", (spec.transfer_id, encoded, spec.size, spec.version, spec.client))

    def _spec(self, transfer_id):
        row = self.db.execute("SELECT spec FROM transfers WHERE id=?", (transfer_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown transfer")
        data = json.loads(row[0])
        data["chunk_sha256"] = tuple(data["chunk_sha256"])
        spec = TransferSpec(**data)
        spec.validate()
        return spec

    def missing(self, transfer_id):
        spec = self._spec(transfer_id)
        present = {row[0] for row in self.db.execute("SELECT position FROM chunks WHERE transfer_id=?", (transfer_id,))}
        return [i for i in range(len(spec.chunk_sha256)) if i not in present]

    def put(self, transfer_id, position, data):
        with self.db:
            self._begin()
            spec = self._spec(transfer_id)
            self._live(spec)
            if type(position) is not int or not 0 <= position < len(spec.chunk_sha256):
                raise ValueError("Invalid chunk position")
            expected_size = min(spec.chunk_bytes, spec.size - position * spec.chunk_bytes)
            if len(data) != expected_size or sha256(data) != spec.chunk_sha256[position]:
                raise ValueError("Chunk checksum or length mismatch")
            existing = self.db.execute("SELECT data FROM chunks WHERE transfer_id=? AND position=?", (transfer_id, position)).fetchone()
            if existing:
                if existing[0] != data:
                    raise ValueError("Conflicting duplicate chunk")
                return False
            self.db.execute("INSERT INTO chunks VALUES (?, ?, ?)", (transfer_id, position, data))
        return True

    def finalize(self, transfer_id):
        with self.db:
            self._begin()
            spec = self._spec(transfer_id)
            previous = self.db.execute("SELECT payload_sha256 FROM eligible WHERE transfer_id=?", (transfer_id,)).fetchone()
            if previous:
                return {"newly_eligible": False, "payload_sha256": previous[0]}
            self._live(spec)
            digest, size, count = hashlib.sha256(), 0, 0
            for row in self.db.execute("SELECT position, data FROM chunks WHERE transfer_id=? ORDER BY position", (transfer_id,)):
                if row["position"] != count or sha256(row["data"]) != spec.chunk_sha256[count]:
                    raise ValueError("Incomplete or corrupted persisted transfer")
                digest.update(row["data"])
                size += len(row["data"])
                count += 1
            if count != len(spec.chunk_sha256) or size != spec.size or digest.hexdigest() != spec.payload_sha256:
                raise ValueError("Incomplete payload or full checksum mismatch")
            try:
                self.db.execute("INSERT INTO eligible VALUES (?, ?, ?, ?)", (transfer_id, spec.version, spec.client, spec.payload_sha256))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Client already has an eligible update at this version") from exc
        return {"newly_eligible": True, "payload_sha256": spec.payload_sha256}

    def eligible_count(self):
        return self.db.execute("SELECT COUNT(*) FROM eligible").fetchone()[0]

    def export(self, transfer_id, path):
        spec = self._spec(transfer_id)
        if not self.db.execute("SELECT 1 FROM eligible WHERE transfer_id=?", (transfer_id,)).fetchone():
            raise ValueError("Only a finalized payload may be exported")
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        checksum = hashlib.sha256()
        size = 0
        with temporary.open("wb") as stream:
            for position, block in self.db.execute("SELECT position, data FROM chunks WHERE transfer_id=? ORDER BY position", (transfer_id,)):
                if position * spec.chunk_bytes != size or sha256(block) != spec.chunk_sha256[position]:
                    raise ValueError("Persisted chunk is corrupt or missing")
                stream.write(block)
                checksum.update(block)
                size += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if size != spec.size or checksum.hexdigest() != spec.payload_sha256:
            temporary.unlink()
            raise ValueError("Exported payload checksum mismatch")
        temporary.replace(path)
        return str(path)
