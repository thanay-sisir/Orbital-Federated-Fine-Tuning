import hashlib
import json
import os
import sqlite3
from pathlib import Path


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class MissionLedger:
    def __init__(self, path, identity):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS commits (round_id INTEGER PRIMARY KEY, state TEXT NOT NULL, adapter BLOB NOT NULL, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS completion (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
        """)
        value = json.dumps(identity, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO identity VALUES (1, ?)", (value,))
        if self.db.execute("SELECT value FROM identity").fetchone()[0] != value:
            self.close()
            raise ValueError("Resume configuration, source, environment or fault plan changed")

    def close(self):
        self.db.close()

    def attempt(self):
        with self.db:
            return self.db.execute("INSERT INTO attempts(status) VALUES ('started')").lastrowid

    def attempt_done(self, attempt, status):
        with self.db:
            self.db.execute("UPDATE attempts SET status=? WHERE id=?", (status, attempt))

    def latest(self):
        row = self.db.execute("SELECT * FROM commits ORDER BY round_id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        state = json.loads(row["state"])
        if hashlib.sha256(row["adapter"]).hexdigest() != state["global_sha256"]:
            raise ValueError("Committed model blob checksum mismatch")
        return state, bytes(row["adapter"])

    def commit(self, round_id, state, adapter, evidence):
        if round_id != state["next_round"] - 1:
            raise ValueError("Round state does not match commit position")
        if hashlib.sha256(adapter).hexdigest() != state["global_sha256"]:
            raise ValueError("Round adapter checksum mismatch")
        values = (round_id, json.dumps(state, sort_keys=True, allow_nan=False), adapter,
                  json.dumps(evidence, sort_keys=True, allow_nan=False))
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            previous = self.db.execute("SELECT * FROM commits WHERE round_id=?", (round_id,)).fetchone()
            if previous:
                if tuple(previous) != values:
                    raise ValueError("Conflicting replay of committed round")
                return False
            latest = self.db.execute("SELECT MAX(round_id) FROM commits").fetchone()[0]
            if (latest is None and round_id != -1) or (latest is not None and round_id != latest + 1):
                raise ValueError("Round commit is not sequential")
            self.db.execute("INSERT INTO commits VALUES (?, ?, ?, ?)", values)
        return True

    def rounds(self):
        return [json.loads(row[0]) for row in self.db.execute("SELECT evidence FROM commits WHERE round_id>=0 ORDER BY round_id")]

    def finished(self):
        row = self.db.execute("SELECT value FROM completion").fetchone()
        return None if row is None else json.loads(row[0])

    def finish(self, value):
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.finished()
            if existing is not None and existing != value:
                raise ValueError("Conflicting completion record")
            self.db.execute("INSERT OR IGNORE INTO completion VALUES (1, ?)", (encoded,))


class FaultInjector:
    def __init__(self, path, plan=None, mode="exit"):
        self.path = Path(path)
        self.plan = plan or []
        self.mode = mode
        valid = {"after_chunk", "after_eligibility", "before_round_commit", "after_round_commit"}
        for entry in self.plan:
            if set(entry) - {"stage", "round", "client", "chunk", "direction"} or entry.get("stage") not in valid:
                raise ValueError("Invalid fault injection plan")
            for key in ("round", "client", "chunk"):
                if key in entry and (type(entry[key]) is not int or entry[key] < 0):
                    raise ValueError("Invalid fault injection position")
            if entry.get("direction", "upload") not in {"upload", "download"}:
                raise ValueError("Invalid fault injection direction")
        if mode not in {"exit", "raise"}:
            raise ValueError("Invalid fault mode")

    def hit(self, stage, **context):
        from .checkpoint import atomic_json, read

        consumed = read(self.path) if self.path.exists() else []
        for index, entry in enumerate(self.plan):
            if index not in [v["index"] for v in consumed] and all(({"stage": stage, **context}).get(k) == v for k, v in entry.items()):
                consumed.append({"index": index, "stage": stage, **context})
                atomic_json(self.path, consumed)
                if self.mode == "exit":
                    os._exit(75)
                raise RuntimeError("Injected coordinator interruption")
