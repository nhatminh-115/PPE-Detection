[![CI/CD Pipeline](https://github.com/nhatminh-115/PPE-Detection/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/nhatminh-115/PPE-Detection/actions/workflows/ci-cd-pipeline.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Terraform](https://img.shields.io/badge/Terraform-1.10+-purple)

# PPE Detection

Real-time, cloud-native PPE compliance monitoring for construction and industrial environments.

Detects PPE violations from live camera streams, sends instant alerts, generates daily regulation-grounded reports, and continuously improves via a human-in-the-loop retraining flywheel.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Flywheel](#2-data-flywheel)
3. [Infrastructure](#3-infrastructure)
4. [Automated Reporting](#4-automated-reporting)
5. [Model Performance](#5-model-performance)
6. [Setup Guide](#6-setup-guide)

---

## 1. System Architecture

![PPE Vision Architecture](docs/PPE_Pipeline_5.png)

| Component | Technology | Role |
|---|---|---|
| Person Detection | YOLO11s + YOLO26l + WBF | Dual-model ensemble, recall-optimized |
| Tracking | ByteTrack + Box EMA | Stable IDs, spatial smoothing |
| PPE Classification | SigLIP SO400M (ONNX FP16) / EfficientNetV2-B0 (ONNX) | Zero-shot large boxes / trained small boxes |
| Pose Estimation | YOLO26m-pose | Head/torso crop localization for SigLIP |
| State Management | Hysteresis FSM + Classify Lock | Alert fatigue suppression, GPU load reduction |
| Compliance Reporting | LangGraph + Groq LLM + RAG ChromaDB | Regulation-cited daily reports |
| Data Flywheel | S3 + Supabase + EC2 Spot | Automated labeling, drift detection, retraining |
| Infrastructure | Terraform + GitHub Actions | Reproducible AWS provisioning, CI/CD |

### 1.1 Core AI Pipeline

**Stage 1 — Detection Ensemble**

YOLO11s (VisDrone fine-tune, aerial) and YOLO26l (COCO, close-range) run in parallel. Detections are fused via WBF (weights=[2,1]) and tracked with ByteTrack + Box EMA (α=0.6). Only persons inside the user-defined polygon zone (`cv2.pointPolygonTest`) proceed to classification.

**Stage 2 — Classify Lock FSM**

Each tracker follows a three-state lifecycle: **ACTIVE** (classify every frame, lock after 3 stable verdicts) → **LOCKED** (cached verdict, inference skipped) → **FORCE RECHECK** (after 20s lock age or confidence drop). Eliminates redundant GPU inference on steady-state workers.

**Stage 3 — Pose-Guided PPE Classification**

Routing by bounding box min-side:

- **< 110px → EfficientNetV2-B0 (ONNX):** Full-body crop + 10% padding. Trained on Ultralytics PPE dataset (F1=0.92). Dynamic batch inference.
- **≥ 110px → SigLIP SO400M (ONNX FP16):** Zero-shot via text prompts. YOLO pose runs every 9 frames to extract head/torso keypoints for precise crops. HSV color priors post-process hardhat probability (white/yellow boost +0.10–0.12, dark penalty −0.14). PyTorch model loaded once for text embedding pre-computation, then offloaded to free ~800 MB RAM.

**Alerting**

- Hysteresis FSM with EMA (α=0.5) controls SAFE → WARN → VIOLATION transitions.
- 3-second per-item accumulation timer; 5-minute per-item cooldown keyed by `(session_id, tracker_id)`.
- Violations written asynchronously to Supabase + Telegram alert in parallel.

### 1.2 Multi-Camera

Each camera has an isolated inference thread with its own tracker, EMA, FSM, and classify lock. A producer-consumer queue decouples frame I/O from GPU inference. Streams served at `/video_feed/{cam_id}`.

---

## 2. Data Flywheel

```
Violations (Supabase)
  └── Second Opinion Agent (Groq Scout — skips already-labeled)
    └── Human Labeling (Label Studio UI)
      └── S3 Export (human-confirmed only)
        └── Drift Monitor
          ├── Scout disagreement (early-warning)
          └── Human-confirmed disagreement (retrain trigger)
            └── EC2 Spot Retrain (EfficientNet → MLflow)
```

- **Scout Agent:** Groq vision LLM (Llama 4 Scout) runs nightly, auto-labels uncertain events as `has-fp / has-fn / all-correct`, writes to `ppe_labels` with `is_auto_labeled=True`.
- **Human Labels:** Label Studio surfaces merged violation events. Crops upscaled with Real-ESRGAN x4 before storage.
- **Drift Trigger:** Retrain fires when 7-day `human_disagreement_rate > 0.25` AND `confirmed_samples >= 100`. EC2 Spot auto-terminates after export.

---

## 3. Infrastructure

| Resource | Description |
|---|---|
| S3 Bucket | Flywheel dataset (versioned, AES-256, lifecycle to IA/Glacier) |
| EC2 Launch Template | t3.medium Spot, Amazon Linux 2023, IMDSv2 |
| IAM Role | S3 read/write + self-terminate |
| Security Group | Outbound-only |

Remote state: S3 at `terraform/state/terraform.tfstate` (Terraform 1.10+ native locking, no DynamoDB).

| Workflow | Trigger | Action |
|---|---|---|
| `ci-cd-pipeline.yml` | Push to `main` | pytest → Docker build/push |
| `daily_report.yml` | 23:00 ICT daily | Scout → report → S3 export → drift monitor → Telegram |
| `retrain_trigger.yml` | 23:30 ICT daily | Drift check → EC2 Spot launch |
| `terraform.yml` | Push/PR on `terraform/**` | plan (PR) / apply (main) |

**Artifact Registry**

| Artifact | Location |
|---|---|
| Docker image | `nhatminh115/ppe_system:latest` |
| EffNet ONNX | MLflow/DagsHub — auto-discovered latest run, stable fallback `af05de89` |
| SigLIP ONNX FP16 | HF Hub — `Nhatminh1234/siglip-so400m-ppe-fp16` (auto-downloaded at startup) |
| Dataset | S3 — `ppe-flywheel/` |

---

## 4. Automated Reporting

Daily at 23:00 ICT: pull violations → Scout second opinion → RAG on Thong tu 25/2022/TT-BLDTBXH (229 pages, 428 chunks, ChromaDB) → Groq LLM (Llama 3.3 70B) narrative → S3 + Telegram + Supabase `daily_reports`.

Reports cite specific Dieu numbers and Phu luc I page entries rather than generic safety advice.

| Criterion | Score |
|---|---|
| Factual Accuracy | 5.0 / 5 |
| Regulation Citation | 5.0 / 5 |
| Specificity | 5.0 / 5 |
| Actionability | 3.9 / 5 |
| Conciseness | 5.0 / 5 |
| **Overall** | **4.8 / 5** |

---

## 5. Model Performance

### Detection (VisDrone val, 548 images)

| Model | Precision | Recall | mAP@50 | FPS |
|---|---|---|---|---|
| YOLO11s (fine-tuned) | 0.760 | 0.611 | 0.684 | 45.8 |
| YOLO26l (pretrained) | 0.590 | 0.356 | 0.418 | 22.3 |
| **WBF Ensemble** | **0.556** | **0.764** | — | ~18* |

*End-to-end with pose model (POSE_EVERY=9) on NVIDIA RTX 5070.

### PPE Classifier

| Model | F1 | Notes |
|---|---|---|
| EfficientNetV2-B0 | 0.92 | Trained, small boxes |
| SigLIP SO400M | — | Zero-shot, large boxes |

### EfficientNetV2-B0 ONNX (CPU)

| Format | Latency | Throughput | Size |
|---|---|---|---|
| PyTorch (CPU) | ~45 ms | ~22 FPS | 22.4 MB |
| ONNX FP32 (CPU) | 7.46 ms | 134 FPS | 22.4 MB |

### SigLIP SO400M ONNX (GPU, batch=2)

| Format | Size | Cosine sim | Latency |
|---|---|---|---|
| PyTorch FP32 | 1632 MB | — | 58.5 ms |
| ONNX FP16 | 816 MB | 0.999997 | 18.2 ms |

FP16 uses native CUDA Tensor Core ops (28 boundary Cast nodes vs 369 CPU fallback nodes in INT8). **3.2x GPU speedup** vs PyTorch FP32.

To regenerate: `python scripts/export_siglip_onnx.py`

---

## 6. Setup Guide

### Required external services

| Service | What you need |
|---|---|
| [Supabase](https://supabase.com) | Project URL, anon key, service role key |
| [DagsHub](https://dagshub.com) | MLflow URI, username, token |
| [Groq](https://console.groq.com) | API key |
| Telegram | Bot token + chat ID |
| AWS | Access key + secret (S3/EC2/IAM) |

### Step 1 — Configure `.env`

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_role_key

MLFLOW_TRACKING_URI=https://dagshub.com/your-username/PPE-Detection.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

GROQ_API_KEY=your_groq_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

S3_BUCKET=your-s3-bucket-name
S3_PREFIX=ppe-flywheel
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

REPORT_USE_LANGGRAPH=1
CLEANUP_SECRET=your_random_secret

# SigLIP ONNX FP16 — auto-downloaded from HF Hub on first startup
SIGLIP_ONNX_ENABLED=1
SIGLIP_ONNX_HF_REPO=Nhatminh1234/siglip-so400m-ppe-fp16
```

### Step 2 — Set up Supabase

Run [`supabase/consolidated_schema.sql`](supabase/consolidated_schema.sql) in the SQL Editor. This creates all five tables: `ppe_labels`, `ppe_label_audit`, `ppe_violations`, `daily_reports`, `drift_log`.

Create a **public** storage bucket named `ppe-crops`.

### Step 3 — Build RAG index (optional)

Only needed when the regulation document changes:

```bash
python rag/ingest.py
```

### Step 4 — Provision AWS infrastructure

```bash
cd terraform
cp backend.hcl.example backend.hcl
# Set your S3 bucket name in backend.hcl
terraform init -backend-config=backend.hcl
terraform apply
```

Save `retrain_launch_template_id` and `retrain_subnet_id` from the outputs for Step 5.

### Step 5 — GitHub Actions secrets

Add to **Settings → Secrets → Actions**: `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.

Variables: `AWS_DEFAULT_REGION`, `S3_PREFIX`, `RETRAIN_LAUNCH_TEMPLATE_ID`, `RETRAIN_SUBNET_ID`.

### Path A — Run from source

```bash
git clone https://github.com/nhatminh-115/PPE-Detection.git
cd PPE-Detection
# place .env here
python main.py
```

Or via Docker:

```bash
docker-compose up --build -d
```

### Path B — Deploy pre-built image

```bash
docker run -d \
  --gpus all \
  --name ppe_inference \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/model_cache:/app/model_cache \
  nhatminh115/ppe_system:latest
```

### Step 6 — Access the dashboard

Open `http://localhost:8000/`.

| Endpoint | Description |
|---|---|
| `/` | Main dashboard |
| `/video_feed/{cam_id}` | Per-camera live stream |
| `/api/violations` | Violation audit log |
| `/api/cameras` | Camera registry |
| `/api/report/generate?date=YYYY-MM-DD` | Trigger daily report |

**Add a camera:**

```bash
# Webcam
curl -X POST "http://localhost:8000/api/cameras/add_webcam?camera_index=0&label=Zone+A"
# RTSP
curl -X POST "http://localhost:8000/api/cameras/add_rtsp?url=rtsp://...&label=Gate+Camera"
```

Or use the dashboard: **Add → Webcam/RTSP → assign label**.

**Define a zone:** Add camera → hover cell → **+ ZONE** → click polygon vertices → **Save Zone**.

| Symptom | Fix |
|---|---|
| CUDA out of memory | Reduce batch; SigLIP FP16 uses ~800 MB less GPU RAM than PyTorch |
| GPU not detected | `docker run --rm --runtime=nvidia nvidia/cuda:12.8-runtime nvidia-smi` |
| Port 8000 in use | Change to `-p 8001:8000` |

**Video Demo**

https://github.com/user-attachments/assets/56d2bce2-6f40-4413-a22e-83af942a0876
