# Signature Detection System

An end-to-end ML system for detecting handwritten signatures in document images and comparing them for authenticity. Built with **Faster R-CNN** (Apache 2.0) — no Ultralytics/AGPL dependencies.

---

## Features

| Feature | Description |
|---|---|
| **Signature Detection** | Locates signatures in documents, returns bounding boxes + confidence |
| **Similarity Score** | Compares two signatures, returns 0.0–1.0 match score |
| **REST API** | FastAPI with 6 endpoints, interactive docs at `/docs` |
| **Async Queue** | Celery + Redis for background job processing |
| **Docker Ready** | API, Worker, and Redis run via `docker-compose up` |
| **AWS Deployment** | ECS Fargate + S3 + SageMaker MLOps pipeline |

---

## Tech Stack

| Layer | Tool | License |
|---|---|---|
| Object Detection | torchvision Faster R-CNN ResNet50 FPN v2 | Apache 2.0 |
| Similarity | torchvision ResNet18 (cosine similarity) | Apache 2.0 |
| API Framework | FastAPI + Uvicorn | MIT |
| Task Queue | Celery + Redis | BSD |
| Containers | Docker + Docker Compose | Apache 2.0 |
| Cloud | AWS ECS, S3, ECR, SageMaker | — |
| Dataset | HuggingFace Hub | Apache 2.0 |

---

## Project Structure

```
signature-detection/
├── app/
│   ├── main.py          # FastAPI app — 6 endpoints
│   ├── detector.py      # Faster R-CNN detection logic
│   ├── similarity.py    # ResNet18 embeddings + cosine similarity
│   └── schemas.py       # Pydantic request/response models
├── worker/
│   ├── worker.py        # Celery + Redis configuration
│   └── tasks.py         # detect_task + similarity_task
├── models/
│   └── best_model.pth   # ← download from Google Drive (not in repo)
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Docker Desktop
- `best_model.pth` placed in `models/` (download link below)

### Option 1 — Python directly

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for interactive API docs.

### Option 2 — Docker Compose (all services)

```bash
docker-compose up --build
```

Starts API (port 8000) + Celery worker + Redis automatically.

---

## API Endpoints

### `GET /health`
Check server status.
```json
{"status": "healthy"}
```

---

### `POST /detect`
Upload an image → get signature bounding boxes.

**Input:** `multipart/form-data` — `file` (image)

**Response:**
```json
{
  "num_signatures": 1,
  "boxes": [
    {
      "x1": 78.9,
      "y1": 427.0,
      "x2": 275.8,
      "y2": 473.1,
      "confidence": 0.9906
    }
  ],
  "message": "Found 1 signature(s)."
}
```

---

### `POST /similarity`
Upload two signature images → get a match score.

**Input:** `multipart/form-data` — `file1` and `file2`

**Response:**
```json
{
  "similarity_score": 0.8923,
  "is_match": true,
  "message": "Signatures match."
}
```

> `is_match` is `true` when score ≥ 0.75

---

### `POST /queue/detect` and `POST /queue/similarity`
Same as above but runs in the background via Celery worker. Returns a `job_id` immediately.

```json
{
  "job_id": "abc123",
  "status": "queued",
  "check_at": "/queue/result/abc123"
}
```

### `GET /queue/result/{job_id}`
Poll for the background job result.

```json
{
  "job_id": "abc123",
  "status": "done",
  "result": { "num_signatures": 1, "boxes": [...] }
}
```

---

## Model

| Property | Value |
|---|---|
| Architecture | Faster R-CNN ResNet50 FPN v2 |
| Pretrained on | COCO (Microsoft) |
| Fine-tuned on | netsol-signature-detection/Signature-Detection-System |
| Training images | ~23,000 |
| Epochs | 5 |
| Batch size | 4 |
| Image size | 416 × 416 |
| Final loss | 0.0850 |
| Test confidence | 0.99 |

### Download Model Weights

The trained `best_model.pth` is not stored in this repo (GitHub 100 MB limit).

Download from Google Drive and place in `models/best_model.pth`:
> Google Drive → `MyDrive/signature_model/best_model.pth`

---

## Dataset

**HuggingFace:** `netsol-signature-detection/Signature-Detection-System`

```python
from huggingface_hub import hf_hub_download

zip_path = hf_hub_download(
    repo_id="netsol-signature-detection/Signature-Detection-System",
    filename="merged_signature_dataset.zip",
    repo_type="dataset"
)
```

Labels are in YOLO format: `class x_center y_center width height` (normalised 0–1).

---

## AWS Architecture

```
User
 └── API Gateway
      └── ECS Fargate
           ├── FastAPI Service  ──→  Redis (ElastiCache)  ──→  Worker Service
           └── (loads model from S3)                              └── (loads model from S3)

HuggingFace upload  ──→  SageMaker  ──→  S3 (new model)  ──→  ECS redeploy
```

| AWS Service | Purpose |
|---|---|
| API Gateway | HTTPS entry point, auth |
| ECS Fargate | Runs API + Worker containers |
| ElastiCache (Redis) | Message queue between API and Worker |
| S3 | Model weights storage |
| ECR | Docker image registry |
| CloudWatch | Logs and metrics |
| SageMaker | Auto-retraining pipeline |

---

## MLOps Pipeline

New dataset uploaded to HuggingFace → webhook triggers AWS Lambda → Lambda starts SageMaker training job → new `best_model.pth` saved to S3 → ECS tasks restart with updated model.

---

## License

This project uses only **Apache 2.0** and **MIT** licensed libraries.  
No AGPL dependencies (Ultralytics/YOLO is not used).
