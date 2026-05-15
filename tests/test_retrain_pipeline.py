"""
Integration tests for the retrain pipeline.

MLFlow test (auto-loads .env):
    pytest tests/test_retrain_pipeline.py -v -s

EC2 test — triggers real GitHub Actions workflow (force=true, min_samples=1):
    pytest tests/test_retrain_pipeline.py::test_ec2_workflow_triggered -v -s
    Then monitor: https://github.com/nhatminh-115/SentinelVision/actions
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_OWNER = "nhatminh-115"
REPO_NAME  = "SentinelVision"


# ── helpers ────────────────────────────────────────────────────────────────────

def _mlflow_creds_present() -> bool:
    return bool(
        os.environ.get("MLFLOW_TRACKING_URI")
        and os.environ.get("MLFLOW_TRACKING_USERNAME")
        and os.environ.get("MLFLOW_TRACKING_PASSWORD")
    )


def _github_token_present() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN"))


# ── MLflow ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _mlflow_creds_present(), reason="MLFlow creds not set")
def test_mlflow_run_created_on_dagshub():
    """Connects to DagsHub MLFlow and creates a real smoke-test run."""
    import mlflow
    from training.retrain_effnet import log_to_mlflow, EXPERIMENT

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = Path(tmp) / "dummy_weights.pt"
        weights_path.write_bytes(b"\x00" * 64)
        os.environ["ROLLING_RATE"] = "smoke-test"
        run_id = log_to_mlflow(str(weights_path), n_train=10, n_val=2)

    assert run_id, "log_to_mlflow() should return a non-empty run_id"

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.data.params.get("trigger") == "drift"
    client.set_tag(run_id, "smoke_test", "true")

    print(f"\n[OK] MLflow run created: {run_id}")
    print(f"     {os.environ['MLFLOW_TRACKING_URI']}#/experiments/6/runs/{run_id}")


@pytest.mark.skipif(not _mlflow_creds_present(), reason="MLFlow creds not set")
def test_mlflow_experiment_exists_after_run():
    """Experiment is auto-created by log_to_mlflow; verify it shows up in the registry."""
    import mlflow
    from training.retrain_effnet import EXPERIMENT

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert experiment is not None, f"Experiment '{EXPERIMENT}' not found on DagsHub"
    print(f"\n[OK] Experiment found: id={experiment.experiment_id} name={experiment.name}")


# ── EC2 via GitHub Actions ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _github_token_present(), reason="GITHUB_TOKEN not set")
def test_ec2_workflow_triggered():
    """
    Triggers retrain_trigger.yml with force=true and min_samples=1.
    Verifies the dispatch was accepted (HTTP 204) then polls until the run appears.
    Actual EC2 launch and self-termination happen inside the workflow.
    """
    import requests
    import time

    token   = os.environ["GITHUB_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    base = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

    # 1. Dispatch the workflow
    resp = requests.post(
        f"{base}/actions/workflows/retrain_trigger.yml/dispatches",
        headers=headers,
        json={
            "ref": "main",
            "inputs": {
                "force":       "true",
                "min_samples": "1",
                "date":        "",
            },
        },
        timeout=15,
    )
    assert resp.status_code == 204, (
        f"Dispatch failed: {resp.status_code} {resp.text}"
    )
    print("\n[OK] Workflow dispatch accepted (204)")

    # 2. Poll for the new run to appear (up to 30 s)
    run_url = None
    for _ in range(10):
        time.sleep(3)
        runs = requests.get(
            f"{base}/actions/workflows/retrain_trigger.yml/runs",
            headers=headers,
            params={"per_page": 1, "branch": "main"},
            timeout=10,
        ).json()
        if runs.get("workflow_runs"):
            latest = runs["workflow_runs"][0]
            run_url = latest["html_url"]
            status  = latest["status"]
            print(f"[OK] Run found: {run_url} (status={status})")
            break

    assert run_url, "Workflow run did not appear within 30s"
    print(f"\nMonitor EC2 after 'retrain-effnet' job starts:")
    print(f"  Workflow : {run_url}")
    print(f"  EC2      : https://console.aws.amazon.com/ec2/v2/home#Instances")


# ── dataset builder (unit, no network) ────────────────────────────────────────

def test_row_to_label_wearing():
    from training.retrain_effnet import _row_to_label
    assert _row_to_label({"human_missing": []}) == {"hardhat": 1, "vest": 1}


def test_row_to_label_missing_hardhat():
    from training.retrain_effnet import _row_to_label
    assert _row_to_label({"human_missing": ["hardhat"]}) == {"hardhat": 0, "vest": 1}


def test_row_to_label_missing_both():
    from training.retrain_effnet import _row_to_label
    assert _row_to_label({"human_missing": ["hardhat", "vest"]}) == {"hardhat": 0, "vest": 0}


def test_row_to_label_none_missing():
    from training.retrain_effnet import _row_to_label
    assert _row_to_label({"human_missing": None}) == {"hardhat": 1, "vest": 1}


# ── workflow YAML sanity ───────────────────────────────────────────────────────

def test_ec2_launch_command_in_workflow():
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/retrain_trigger.yml"
    content  = workflow.read_text()
    assert "aws ec2 run-instances" in content
    assert "LaunchTemplateId" in content
    assert "terminate-instances" in content
