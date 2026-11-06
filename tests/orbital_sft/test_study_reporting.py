import importlib.util
import json
import sys

import pytest

from swarmshield.orbital.config import project_path


def reporting_module():
    spec = importlib.util.spec_from_file_location("orbital_study_report", project_path("scripts/orbital_study_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receipt_metrics_distinguish_payload_wire_and_wait():
    report = reporting_module()
    receipt = {"direction": "upload", "complete": True, "ready": 10, "arrival": 16, "verified_bytes": 100, "chunks": [{"start": 12, "finish": 15, "wire_bytes": 110}]}
    result = report.communication([receipt])
    assert result["wire_bytes"] == 110
    assert result["verified_payload_bytes"] == 100
    assert result["wait_before_first_chunk_seconds"] == 2
    assert result["transfer_elapsed_seconds"] == 6
    assert result["station_service_seconds"] == 3


def test_gpu_cost_counts_allocation_once_not_job_steps(monkeypatch):
    report = reporting_module()

    class Result:
        stdout = "JobIDRaw|JobName|State|ExitCode|ElapsedRaw|AllocTRES|TotalCPU|MaxRSS\n1|train|COMPLETED|0:0|1800|billing=100,cpu=20,gres/gpu=5|00:10:00|\n1.0|step|COMPLETED|0:0|1800|gres/gpu=5||\n"

    monkeypatch.setattr(report.subprocess, "run", lambda *args, **kwargs: Result())
    values, _ = report.accounting([1])
    assert len(values) == 1
    assert values[0]["allocated_gpu_hours"] == pytest.approx(2.5)
    assert values[0]["slurm_billing_weight_hours"] == pytest.approx(50)


def test_missing_evaluation_is_reported_as_incomplete(tmp_path, monkeypatch):
    report = reporting_module()
    plan = {"jobs": [], "runs": {"final_evaluation": "missing-run"}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(report, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(report, "digest", lambda value: "test-only-hash")
    monkeypatch.setattr(report, "accounting", lambda jobs: ([], ""))
    monkeypatch.setattr(sys, "argv", ["report", "--plan", "plan.json", "--output", "report-output"])
    report.main()
    result = json.loads((tmp_path / "report-output/study.json").read_text())
    assert result["incomplete_runs"] == ["final_evaluation"]
    assert "no outcomes are inferred" in (tmp_path / "report-output/report.md").read_text()


def test_control_tensor_hash_detects_changed_initialization(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    report = reporting_module()
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    save_file({"lora_A": np.array([1, 2], dtype=np.float32)}, str(first))
    save_file({"lora_A": np.array([1, 3], dtype=np.float32)}, str(second))
    assert report.tensor_hash(first) != report.tensor_hash(second)
    assert report.tensor_hash(first) == report.tensor_hash(first)
