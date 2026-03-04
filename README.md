[![CI/CD Pipeline](https://github.com/nhatminh-115/PPE-Detection/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/nhatminh-115/PPE-Detection/actions/workflows/ci-cd-pipeline.yml)

# PPE Detection

PPE Vision is a real-time, cloud-native inference engine designed for autonomous Personal Protective Equipment (PPE) compliance monitoring. Engineered for high-stakes construction and industrial environments, the system decouples complex Deep Learning inference from robust Data Observability and State Management.

---

## 1. System Architecture Overview

The platform is designed around a microservices architecture, exposing a scalable FastAPI application that integrates natively with MLflow for dynamic artifact retrieval and Supabase for unstructured event logging.

### 1.1 Core AI Pipeline

**Stage 1: Spatial-Temporal Detection Ensemble**
To mitigate domain shift between varying camera angles (aerial vs. close-circuit), the system implements a dual-YOLO ensemble:
* **Primary Detector:** Fine-tuned on VisDrone for dense, small-scale pedestrian representations.
* **Secondary Detector:** COCO-pretrained (`yolo26l.pt`) for high-fidelity close-range features.
* **Fusion & Tracking:** Detections are fused using Weighted Boxes Fusion (WBF) to suppress redundant overlapping. Temporal continuity is maintained via ByteTrack coupled with a Box Exponential Moving Average (EMA) to eliminate spatial jittering.

**Stage 2: Classification**
Person tensors are classified using an **EfficientNetV2-B0** multi-label network. To ensure interpretability and suppress False Positives (e.g., misclassifying gray fabric as hardhats), we implemented a **Forward-CAM (Class Activation Map)** intercepting the `timm` feature extraction layer.

The activated spatial regions are subjected to a **Dual-Channel HSV Color Penalizer**:
* **Value (Brightness) Penalty:** Regions with $V < 70$ receive a probability penalty (filtering dark hair/shadows).
* **Saturation Penalty:** Regions with $S < 80$ receive a small penalty (filtering gray concrete/clothing).

### 1.2 State Management & Observability

To prevent alert fatigue and database spamming caused by transient model uncertainty:
* **Hysteresis Finite State Machine (FSM):** Controls state transitions (SAFE, WARN, VIOLATION) with a strict margin.
* **Temporal Accumulation:** A violation must be sustained for a few seconds before triggering an event.
* **Idempotent Event Logging:** Once validated, the violation is logged asynchronously (Fire-and-Forget ThreadPool) to a **Supabase PostgreSQL** instance using a flexible `JSONB` schema, accompanied by a localized crop image blob for manual auditing.

---

## 2. Infrastructure

The infrastructure follows GitOps principles, ensuring the inference environment is fully version-controlled, reproducible, and seamlessly integrated with model training lifecycles.

* **Artifact Fetching:** The API dynamically fetches versioned `.pt` artifacts directly from the MLflow DagsHub Registry via uniquely identifiable `run_id`s, utilizing a Local Cache collision-avoidance mechanism.
* **Containerization:** The monolithic engine is packaged in a Docker image. Deployment via `docker-compose.yml` mounts persistent volumes for the model cache and blob storage, enabling hardware-accelerated NVIDIA GPU Passthrough.
* **Continuous Integration (CI):** GitHub Actions strictly governs the repository. The pipeline automatically provisions a mock environment and executes `pytest` using `unittest.mock` to validate API logic and endpoint integrity prior to Docker registry pushes.

---

## Results and Images

**EfficientNet Training Metrics**

<img width="2183" height="1105" alt="image" src="https://github.com/user-attachments/assets/3b3510ac-ed98-4ed1-ba45-359ae912a773" />

**Yolo11s Training Metrics**

<img width="2183" height="1105" alt="image" src="https://github.com/user-attachments/assets/83d2b745-9dc0-450c-af28-bdfd9b73d1dd" />

**Supabase Database**

<img width="2559" height="1395" alt="image" src="https://github.com/user-attachments/assets/f1cdcb07-20a2-4b91-b62e-58f70959e640" />


**User Interface**

<img width="2559" height="1462" alt="image" src="https://github.com/user-attachments/assets/b4eb1f13-f5f4-478f-b4c8-21d3a891af9f" />

<img width="2559" height="1459" alt="image" src="https://github.com/user-attachments/assets/212b08e1-764d-460b-a091-cb2cb08f9c53" />

**Video Demo**


https://github.com/user-attachments/assets/7d18e0cb-dba9-4d97-8319-41e1005efcc8


---

## 4. Deployment Guide

### Prerequisites
* Docker Engine 24.0+ & Docker Compose
* NVIDIA Container Toolkit (for GPU inference)
* Python 3.11+ (for local execution)

### Quick Start

1.  **Clone & Configure:**
    ```bash
    git clone [https://github.com/nhatminh-115/PPE-Detection.git](https://github.com/nhatminh-115/PPE-Detection.git)
    cd PPE-Detection
    ```
    Create a `.env` file containing your production keys:
    ```env
    SUPABASE_URL=[https://your-project.supabase.co](https://your-project.supabase.co)
    SUPABASE_KEY=your_anon_key
    MLFLOW_TRACKING_URI=[https://dagshub.com/nhatminh-115/PPE-Detection.mlflow](https://dagshub.com/nhatminh-115/PPE-Detection.mlflow)
    ```

2.  **Provision Infrastructure:**
    ```bash
    docker-compose up --build -d
    ```

3.  **Access Telemetry:**
    * **Dashboard SPA:** `http://localhost:8000/`
    * **Inference Stream:** `http://localhost:8000/video_feed`
    * **Audit Logs:** `http://localhost:8000/api/violations`

---

## 5. License
MIT License - See `LICENSE` for details.
