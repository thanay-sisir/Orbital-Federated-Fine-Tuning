import hashlib
import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[3]


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes project: {value}")
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Model(Strict):
    identity: str = "Qwen/Qwen3.8-27B"
    revision: str = "REQUIRED_COMMIT_SHA"
    tokenizer_revision: str = "REQUIRED_COMMIT_SHA"
    path: str = "artifacts/model"
    loader: str = "AutoModelForImageTextToText"
    target_modules: list[str] = Field(default_factory=list)
    rank: int = Field(8, ge=1, le=128)
    alpha: int = Field(16, ge=1)
    quantization: str = "nf4"
    dtype: str = "bfloat16"
    factor_policy: str = "fixed_A_train_B"


class Data(Strict):
    identity: str = "allenai/tulu-3-sft-mixture"
    revision: str = "REQUIRED_COMMIT_SHA"
    raw_path: str = "artifacts/raw/tulu.jsonl"
    prepared_path: str = "artifacts/prepared/pilot"
    source_terms: dict[str, str] = Field(default_factory=dict)
    licence_review: str = "REQUIRED_REVIEW"
    max_examples: int = Field(10000, ge=10)
    max_length: int = Field(1024, ge=16)
    validation_fraction: float = Field(0.05, gt=0, lt=0.25)
    test_fraction: float = Field(0.05, gt=0, lt=0.25)
    partition: str = "source_skew"
    thinking: bool = False
    truncation: str = "drop"


class Training(Strict):
    run_seed: int | None = Field(None, ge=0, le=2147483647)
    clients: int = Field(8, ge=1, le=8)
    workers: int = Field(1, ge=1, le=8)
    rounds: int = Field(3, ge=1)
    local_steps: int = Field(4, ge=1)
    full_shard_pass: bool = False
    epoch_number: int = Field(1, ge=1, le=3)
    parent_run: str | None = None
    evaluate_test: bool = True
    validation_examples: int = Field(16, ge=1)
    accumulation: int = Field(4, ge=1)
    learning_rate: float = Field(0.0001, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    grad_clip: float = Field(1.0, gt=0)
    optimizer_reset: str = "each_round"
    schedule: str = "constant"
    device_step_seconds: float = Field(60.0, gt=0)
    device_profile: str = "UNCALIBRATED sensitivity assumption; seconds per optimizer step"
    compatibility_run: str = "results/orbital/compatibility-pilot"
    reload_tolerance: float = Field(0.001, gt=0)


class Station(Strict):
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    elevation_m: float = 0


class Orbit(Strict):
    epoch: str = "2026-11-02T00:00:00+00:00"
    altitude_km: float = Field(550, ge=200, le=2000)
    inclination_deg: float = Field(53, ge=0, le=180)
    planes: int = Field(2, ge=1)
    mission_seconds: float = Field(86400, gt=0, le=604800)
    elevation_deg: float = Field(10, ge=0, lt=90)
    stations: list[Station]


class Network(Strict):
    rate_bps: float = Field(10000000, gt=0)
    chunk_bytes: int = Field(1048576, ge=1)
    framing_bytes: int = Field(64, ge=0)
    ack_bytes: int = Field(32, ge=0)
    acquisition_seconds: float = Field(1, ge=0)
    queue_bytes: int = Field(1073741824, ge=1)
    byte_budget: int = Field(100000000000, ge=1)
    policy: str = "fixed"
    round_seconds: float = Field(14400, gt=0)
    min_participants: int = Field(2, ge=1)


class Settings(Strict):
    schema_version: int = 1
    seed: int = Field(42, ge=0)
    model: Model
    data: Data
    training: Training
    orbit: Orbit
    network: Network

    @property
    def training_seed(self):
        return self.seed if self.training.run_seed is None else self.training.run_seed

    @model_validator(mode="after")
    def check(self):
        if self.training.parent_run is not None:
            project_path(self.training.parent_run)
        if self.training.epoch_number > 1 and not self.training.full_shard_pass:
            raise ValueError("Epoch continuation requires full shard passes")
        if self.training.full_shard_pass and (
            self.training.rounds != 1
            or self.network.min_participants != self.training.clients
        ):
            raise ValueError("A full shard pass requires one round and every client to aggregate")
        if self.training.workers > self.training.clients:
            raise ValueError("Workers exceed virtual clients")
        if self.network.min_participants > self.training.clients:
            raise ValueError("Minimum participants exceeds clients")
        if self.orbit.planes > self.training.clients:
            raise ValueError("Orbital planes cannot exceed virtual clients")
        if len({s.name for s in self.orbit.stations}) != len(self.orbit.stations):
            raise ValueError("Station names must be unique")
        if not self.orbit.stations:
            raise ValueError("At least one station required")
        if self.network.policy not in {"fixed", "contact_aware", "connected", "feasible_fair", "retry_backoff"}:
            raise ValueError("Unsupported scheduler")
        if self.data.partition not in {"iid", "source_skew"}:
            raise ValueError("Unsupported partition")
        if (self.model.factor_policy, self.model.quantization, self.model.dtype) != (
            "fixed_A_train_B",
            "nf4",
            "bfloat16",
        ):
            raise ValueError("Reference implements fixed-A NF4 BF16 only")
        if self.training.optimizer_reset != "each_round" or self.training.schedule != "constant":
            raise ValueError("Reference resets AdamW each round with a constant learning rate")
        if self.data.truncation != "drop":
            raise ValueError("Reference drops overlength conversations")
        for value in [
            self.model.path,
            self.data.raw_path,
            self.data.prepared_path,
            self.training.compatibility_run,
        ]:
            project_path(value)
        return self

    def ready(self, training=False):
        for revision in (self.model.revision, self.model.tokenizer_revision, self.data.revision):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Pin model, tokenizer and dataset to exact 40-character commits")
        if not self.model.target_modules:
            raise ValueError("Review language-only full target-module names before execution")
        if not self.data.source_terms or self.data.licence_review == "REQUIRED_REVIEW":
            raise ValueError("Explicit source allowlist and licence review required")
        project_path(self.data.licence_review).read_text()
        if training:
            manifest = project_path(self.model.path) / "snapshot_manifest.json"
            info = json.loads(manifest.read_text())
            if info["identity"] != self.model.identity or info["revision"] != self.model.revision:
                raise ValueError("Staged model identity mismatch")
            if info["tokenizer_revision"] != self.model.tokenizer_revision:
                raise ValueError("Tokenizer identity mismatch")
            for name, checksum in info["sha256"].items():
                path = (manifest.parent / name).resolve()
                if not path.is_relative_to(manifest.parent.resolve()) or digest(path) != checksum:
                    raise ValueError(f"Staged model checksum mismatch: {name}")
            raw = project_path(self.data.raw_path)
            raw_manifest = raw.with_suffix(".manifest.json")
            if not raw.is_file() or not raw_manifest.is_file():
                raise ValueError("Pinned dataset has not been staged")
            data_info = json.loads(raw_manifest.read_text())
            if (data_info.get("identity"), data_info.get("revision"), data_info.get("sha256")) != (
                self.data.identity,
                self.data.revision,
                digest(raw),
            ):
                raise ValueError("Staged dataset identity or checksum mismatch")
            if data_info.get("source_terms") != self.data.source_terms:
                raise ValueError("Staged dataset source allowlist mismatch")

    def fingerprint(self):
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load(path):
    return Settings.model_validate(yaml.safe_load(project_path(path).read_text()))
