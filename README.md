[![CI/CD Pipeline](https://github.com/nhatminh-115/PPE-Detection/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/nhatminh-115/PPE-Detection/actions/workflows/ci-cd-pipeline.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Terraform](https://img.shields.io/badge/Terraform-1.10+-purple)

# PPE Detection

PPE Vision is a real-time, cloud-native inference engine for autonomous Personal Protective Equipment (PPE) compliance monitoring. Engineered for high-stakes construction and industrial environments, the system integrates a multi-stage deep learning pipeline with a compliance reporting layer grounded in Vietnamese labor regulations.

The system is designed as a closed-loop flywheel: every violation detected feeds into a human labeling interface, disagreements are auto-surfaced by a vision second-opinion agent, labeled data is versioned to S3, and model drift automatically triggers EfficientNet retraining on an EC2 Spot Instance — all without manual intervention.

---

## 1. System Architecture Overview

![PPE Vision Architecture](docs/PPE_Pipeline_5.png)

| Component | Technology | Role |
|---|---|---|
| Person Detection | YOLO11s + YOLO26l + WBF | Dual-model ensemble, recall-optimized |
| Tracking | ByteTrack + Box EMA | Stable ID assignment, spatial smoothing |
| PPE Classification | SigLIP SO400M / EfficientNetV2-B0 | Zero-shot (large) vs trained (small), routed by box size |
| Pose Estimation | YOLO26m-pose | Precise head/torso crop localization for SigLIP |
| State Management | Hysteresis FSM + Classify Lock | Alert fatigue suppression, GPU load reduction |
| Compliance Reporting | LangGraph + Groq LLM + RAG ChromaDB | Regulation-cited daily reports |
| Data Flywheel | S3 + Supabase + EC2 Spot | Automated labeling, drift detection, retraining |
| Infrastructure | Terraform + GitHub Actions | Reproducible AWS provisioning, CI/CD |

### 1.1 Core AI Pipeline

**Stage 1: Spatial-Temporal Detection Ensemble**

To mitigate domain shift between varying camera angles (aerial vs. close-circuit), the system implements a dual-YOLO ensemble alongside a dedicated pose estimator:

* **Primary Detector:** YOLO11s fine-tuned on VisDrone for dense, small-scale pedestrian detection.
* **Secondary Detector:** COCO-pretrained `yolo26l.pt` for high-fidelity close-range detection.
* **Fusion & Tracking:** Detections are fused via Weighted Boxes Fusion (WBF, weights=[2,1]) to suppress redundant overlaps. ByteTrack assigns stable tracker IDs across frames, coupled with Box EMA (α=0.6) to eliminate spatial jittering.

**Zone-Based Spatial Filtering**

Before classification, detections are filtered through a user-defined polygon zone. Each person's foot-point (bottom-center of bounding box) is tested via `cv2.pointPolygonTest`. Only persons inside the zone are passed downstream — eliminating irrelevant detections from background areas.

**Stage 2: Classify Lock FSM**

To reduce redundant GPU inference, each tracker follows a three-state lifecycle:

* **ACTIVE (0-20s):** Classify every detection frame. Requires 3 consecutive stable verdicts before locking.
* **LOCKED:** Cached verdict is reused. Inference is skipped entirely.
* **FORCE RECHECK:** Triggered after 20s of lock age, or when verdict confidence drops below threshold.

**Stage 3: Pose-Guided PPE Classification**

Classification is routed by bounding box size:

* **Small boxes (min side < 110px) → EfficientNetV2-B0:** Full-body crop with 10% padding. Trained on processed Ultralytics PPE dataset (F1=0.92). Lightweight and fast for distant detections.

* **Large boxes (min side >= 110px) → SigLIP SO400M:** Zero-shot classification using text prompts. YOLO26m-pose runs every 9 frames to extract COCO keypoints and define precise crop regions:
  * **Hardhat crop:** Head keypoints (nose, eyes, ears — indices 0-4), pad 45%.
  * **Vest crop:** Torso keypoints (shoulders, hips — indices 5, 6, 11, 12), pad 25%.
  * **Fallback:** Fixed-percentage crop (top 42% for head, 20-85% for torso) when pose confidence is insufficient.

  After SigLIP inference, **HSV Color Priors** post-process the hardhat probability:
  * White pixel ratio (S<=58, V>=148) in head ROI → +0.12 boost
  * Bright pixel ratio (white or yellow) → +0.10 boost
  * Dark pixel ratio (V<=72, ratio>=22%) → -0.14 penalty

**HuggingFace Spaces demo:** [![HuggingFace Spaces](https://img.shields.io/badge/-Live%20Demo-yellow)](https://huggingface.co/spaces/Nhatminh1234/ppe-classifier)

### 1.2 State Management & Observability

* **Hysteresis FSM:** Controls SAFE → WARN → VIOLATION transitions with strict margins to prevent oscillation. EMA smoothing (α=0.5) is applied to raw classification probabilities before state evaluation.

* **Per-Item Temporal Accumulation:** Each PPE item (hardhat, vest) maintains an independent 3-second accumulation timer. A timer increments only while that specific item is in the VIOLATION state (state=0), and resets to zero the moment the item recovers.

* **Per-Item Violation Cooldown:** A 5-minute cooldown is tracked independently per PPE item per camera session. Cooldown state is keyed by `(session_id, tracker_id)` to enforce strict cross-camera isolation.

  Recovery semantics are strict by design:
  * **Full recovery** (all PPE items back to state=2, green) clears all per-item cooldowns for that tracker.
  * **Partial recovery** (e.g. vest restored, hardhat still missing) does NOT reset the hardhat cooldown.
  * **WARN state** (state=1) is explicitly not a recovery condition.

* **Idempotent Event Logging:** Violations are logged asynchronously (Fire-and-Forget ThreadPool) to Supabase PostgreSQL with a JSONB schema, alongside a localized crop image for manual auditing.

* **Instant Telegram Alert:** Each confirmed violation dispatches a Telegram notification with the crop image attached, in parallel with the Supabase insert.

### 1.3 Multi-Camera Architecture

* Each camera is registered dynamically via **CameraRegistry** and assigned a unique `cam_id` and `session_id`.
* Every camera spawns a dedicated **inference thread** with isolated tracker, EMA, FSM, and classify lock state.
* A **producer-consumer pipeline** decouples frame I/O from GPU inference via bounded queues.
* Streams are served at `/video_feed/{cam_id}` and composited into a responsive grid in the UI.
* **Zone polygons** are stored per-camera and persist across stream restarts.

---

## 2. Data Flywheel

The system implements a closed-loop retraining pipeline that continuously improves EfficientNet from real production data.

```
Violations (Supabase)
    └── Human Labeling (Label Studio UI)
            └── Second Opinion Agent (Groq Scout)
                    └── S3 Export (versioned crops + manifests)
                            └── Drift Monitor (7-day rolling disagreement rate)
                                    └── EC2 Spot Retrain (EfficientNet fine-tune → MLflow)
```

### 2.1 Human-in-the-Loop Labeling

The Label Studio tab in the dashboard surfaces merged violation events (keyed by `session_id:tracker_id`) as label cards. Annotators confirm or reject model predictions; verdicts are derived server-side as TP/FP/FN/TN per PPE item. Crops are upscaled with Real-ESRGAN x4 (74ms/crop, +38% sharpness recovery) before storage.

### 2.2 Vision Second Opinion Agent

A Groq vision agent (Llama 4 Scout) runs nightly on all violation events. It auto-labels events as `has-fp`, `has-fn`, or `all-correct` — writing to `ppe_labels` with `is_auto_labeled=True`. Disagreements surface as pre-labeled cards in Label Studio for human review and become the highest-value retraining candidates.

### 2.3 S3 Dataset Versioning

Confirmed human labels and their crops are exported to S3 daily:

```
ppe-flywheel/
  crops/dt=YYYY-MM-DD/<sha256>.jpg     # deduplicated by SHA-256
  labels/dt=YYYY-MM-DD/labels.jsonl    # label manifest
  terraform/state/terraform.tfstate    # Terraform remote state
```

### 2.4 Drift Monitor & Retrain Trigger

The drift monitor computes a daily disagreement rate from Scout auto-labels:

```
disagreement_rate = (has-fp + has-fn) / total auto-labels per day
```

Retrain fires when **both** conditions are met:
- 7-day rolling average `disagreement_rate > DRIFT_THRESHOLD` (default 0.25)
- Total confirmed human labels `>= MIN_CONFIRMED_SAMPLES` (default 100)

Results are logged to the `drift_log` Supabase table. When triggered, a GitHub Actions workflow launches an EC2 Spot Instance (t3.medium) that runs `training/retrain_effnet.py` and self-terminates on completion.

---

## 3. Infrastructure

### 3.1 AWS Resources (Terraform-managed)

All AWS resources are defined in `terraform/` and provisioned via Terraform >= 1.10.

| Resource | Description |
|---|---|
| S3 Bucket | Flywheel dataset storage (versioned, AES-256, lifecycle to S3-IA/Glacier) |
| EC2 Launch Template | t3.medium Spot, Amazon Linux 2023, IMDSv2 enabled |
| IAM Role | Instance profile with S3 read/write and self-terminate permissions |
| Security Group | Outbound-only (Docker pull, S3, MLflow, Supabase) |

Remote state is stored in the same S3 bucket at `terraform/state/terraform.tfstate` with native S3 locking (no DynamoDB required, Terraform 1.10+).

### 3.2 CI/CD

| Workflow | Trigger | Action |
|---|---|---|
| `ci-cd-pipeline.yml` | Push to main | Run pytest, build and push Docker image to Docker Hub |
| `daily_report.yml` | 23:00 ICT daily | Second opinion → S3 export → drift monitor → LangGraph report → Telegram |
| `retrain_trigger.yml` | 23:30 ICT daily | Check drift signal → launch EC2 Spot Instance if triggered |
| `terraform.yml` | Push/PR on `terraform/**` | Terraform plan (PR) / apply (main) |

### 3.3 Artifact Registry

* **Docker:** Docker Hub (`nhatminh115/ppe_system:latest`)
* **Model weights:** MLflow on DagsHub ([tracking URI](https://dagshub.com/nhatminh-115/PPE-Detection.mlflow))
* **Dataset:** S3 (`ppe-flywheel/ppe-flywheel/`)

---

## 4. Automated Reporting & Compliance Intelligence

An automated reporting pipeline runs daily at 23:00 ICT via GitHub Actions cron.

**Pipeline:** Pull violation logs → Second Opinion Agent → S3 export → Drift monitor → RAG lookup on Thong tu 25/2022 → Groq LLM generates narrative → Telegram + Supabase.

**RAG Layer:** The knowledge base indexes 229 pages of Thong tu 25/2022/TT-BLDTBXH (428 chunks) into ChromaDB using `paraphrase-multilingual-MiniLM-L12-v2` embeddings. Violation types detected automatically generate Vietnamese queries, retrieve top-k regulation chunks, and inject them as grounded context into the Groq prompt.

**Result:** Daily reports cite specific Dieu numbers and Phu luc I page entries rather than generic safety advice.

| Component | Technology |
|---|---|
| LLM | Groq API (Llama 3.3 70B) |
| Orchestration | LangGraph 4-agent pipeline |
| Vector Store | ChromaDB (persistent, cosine) |
| Embedding Model | paraphrase-multilingual-MiniLM-L12 |
| Scheduler | GitHub Actions cron (16:00 UTC) |
| Notification | Telegram Bot API |
| Storage | Supabase `daily_reports` table |

**Daily Report (Supabase):**
![Daily Report Supabase](docs/Daily_report_supabase.png)

**Telegram Report:**
![Telegram Report](docs/Telegram_report.png)

---

## 5. Model Performance

### Stage 1: Detection Ensemble (VisDrone val set, 548 images)

| Model | Precision | Recall | mAP@50 | FPS |
|---|---|---|---|---|
| YOLO11s (fine-tuned) | 0.760 | 0.611 | 0.684 | 45.8 |
| YOLO26l (pretrained) | 0.590 | 0.356 | 0.418 | 22.3 |
| **WBF Ensemble** | **0.556** | **0.764** | - | ~18* |

*End-to-end throughput with pose model (POSE_EVERY=9) on NVIDIA RTX 5070.

> Ensemble rationale: Higher recall (0.764 vs 0.611) prioritizes zero missed violations. Residual false positives are suppressed downstream by the classification stage.

### Stage 2: PPE Classifier (Ultralytics PPE dataset)

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| EfficientNetV2-B0 | 0.97 | 0.88 | 0.92 | Trained, handles small boxes |
| SigLIP SO400M | - | - | - | Zero-shot, handles large boxes |

### ONNX Export (EfficientNetV2-B0, CPU deployment)

| Format | Latency | Throughput | Size |
|---|---|---|---|
| PyTorch (CPU) | ~45 ms | ~22 FPS | 22.4 MB |
| ONNX FP32 (CPU) | 7.46 ms | 134 FPS | 22.4 MB |

> ~6x speedup via ONNX Runtime graph optimization. Model artifact: [MLflow Run](https://dagshub.com/nhatminh-115/PPE-Detection.mlflow)

### LLM Report Quality

Evaluated across 10 synthetic violation scenarios via LLM-as-judge (Gemini 3.1 Pro).

| Criterion | Score |
|---|---|
| Factual Accuracy | 5.0 / 5 |
| Regulation Citation | 5.0 / 5 |
| Specificity | 5.0 / 5 |
| Actionability | 3.9 / 5 |
| Conciseness | 5.0 / 5 |
| **Overall** | **4.8 / 5** |

**EfficientNet Training Metrics**
![EfficientNet Metrics](docs/EfficientNet_metrics.png)

**YOLO11s Training Metrics**
![YOLO11s Metrics](docs/Yolo11s_metrics.png)

**Supabase Database**
![Supabase](docs/Supabase.png)

**User Interface**
![UI 1](docs/UI_1.png)
![UI 2](docs/UI_2.png)

**Video Demo**

https://github.com/user-attachments/assets/7d18e0cb-dba9-4d97-8319-41e1005efcc8

---

## 6. Setup Guide

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Docker + Docker Compose | 24.0+ | Run inference engine |
| NVIDIA Container Toolkit | latest | GPU passthrough |
| CUDA driver | 12.8+ | GPU inference |
| Python | 3.11+ | Local scripts |
| Terraform | 1.10+ | AWS provisioning |
| AWS CLI | 2.x | AWS authentication |

### Step 1 — Clone & configure environment

```bash
git clone https://github.com/nhatminh-115/PPE-Detection.git
cd PPE-Detection
```

Create a `.env` file (copy from `.env.example` if available):

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_role_key

# MLflow (DagsHub)
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/PPE-Detection.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# LLM & reporting
GROQ_API_KEY=your_groq_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Data flywheel
S3_BUCKET=your-s3-bucket-name
S3_PREFIX=ppe-flywheel
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Feature flags
REPORT_USE_LANGGRAPH=1
SECOND_OPINION_THRESHOLD=1.0
MUTE_THIRD_PARTY_STARTUP_LOGS=1
QUIET_GET_ACCESS_LOGS=1
```

### Step 2 — Supabase setup

Run SQL migrations in order in the Supabase SQL editor:

```
supabase/migrations/001_ppe_labels.sql
supabase/migrations/002_ppe_labels_sha256.sql
supabase/migrations/003_daily_reports_phase2.sql
supabase/migrations/004_ppe_labels_auto.sql
supabase/migrations/005_drift_log.sql
```

Create a public Storage bucket named `ppe-crops` in Supabase Storage.

### Step 3 — Build the RAG index

```bash
python rag/ingest.py
```

This creates `data/chroma_db`, used by the daily regulation-cited report pipeline.

### Step 4 — Provision AWS infrastructure (Terraform)

Install Terraform >= 1.10 from [developer.hashicorp.com/terraform/install](https://developer.hashicorp.com/terraform/install) and configure AWS CLI:

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, region (us-east-1), leave output blank
```

Initialize and apply:

```bash
cd terraform
cp backend.hcl.example backend.hcl
# Edit backend.hcl: set bucket (your S3 bucket name) and region
terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

After apply, note the outputs:

```bash
terraform output retrain_launch_template_id
terraform output retrain_subnet_id
```

### Step 5 — Configure GitHub Actions secrets and variables

In your GitHub repo → **Settings → Secrets and variables → Actions**:

**Secrets** (sensitive values):

| Secret | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon key |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI |
| `MLFLOW_TRACKING_USERNAME` | MLflow/DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | MLflow/DagsHub token |
| `GROQ_API_KEY` | Groq API key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat ID |
| `S3_BUCKET` | S3 bucket name |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

**Variables** (non-sensitive):

| Variable | Description | Default |
|---|---|---|
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `S3_PREFIX` | S3 key prefix | `ppe-flywheel` |
| `RETRAIN_LAUNCH_TEMPLATE_ID` | From `terraform output` | - |
| `RETRAIN_SUBNET_ID` | From `terraform output` | - |

### Step 6 — Run the inference engine

```bash
docker-compose up --build -d
```

Or pull the pre-built image:

```bash
docker pull nhatminh115/ppe_system:latest
docker-compose up -d
```

### Step 7 — Access the dashboard

| Endpoint | Description |
|---|---|
| `http://localhost:8000/` | Main dashboard |
| `http://localhost:8000/video_feed/{cam_id}` | Per-camera stream |
| `http://localhost:8000/api/violations` | Audit log |
| `http://localhost:8000/api/labels` | Label store |
| `http://localhost:8000/api/second_opinion/run?date=YYYY-MM-DD` | Trigger second opinion |
| `http://localhost:8000/api/report/generate?date=YYYY-MM-DD` | Trigger report |
| `http://localhost:8000/api/cameras` | Camera registry |

### Multi-Camera Setup

```bash
# Add cameras via API
curl -X POST "http://localhost:8000/api/cameras/add_webcam?camera_index=0&label=Zone+A"
curl -X POST "http://localhost:8000/api/cameras/add_rtsp?url=rtsp://...&label=Gate+Camera"

# List active cameras
curl "http://localhost:8000/api/cameras"
```

Or use the dashboard: **Add** → select Webcam or RTSP → assign label.

### Zone Definition

1. Add a camera and start the stream.
2. Hover over a camera cell → click **+ ZONE**.
3. The stream pauses on a frozen frame.
4. Click to place polygon vertices; click the first point to close.
5. Click **Save Zone** — inference resumes and only persons within the zone are checked.

---

## 7. Related Work

* **[Cali Housing MLOps: From Manual to GitOps Architecture](https://github.com/nhatminh-115/cali-housing-mlops)** — IaC and GitOps pipelines (Terraform, AWS EC2, closed-loop CT) forming the foundation for the infrastructure layer above.
