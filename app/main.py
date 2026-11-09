from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import DetectionResponse, BoundingBox, SimilarityResponse
from app.detector import detect_signatures
from app.similarity import compute_similarity

app = FastAPI(
    title="Signature Detection API",
    description="Detects signatures in documents and compares them for similarity.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "Signature Detection API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Synchronous endpoints (instant response) ──────────────────────────────────

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Upload one image → get signature bounding boxes immediately.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    boxes_raw = detect_signatures(image_bytes)
    boxes = [BoundingBox(**b) for b in boxes_raw]

    return DetectionResponse(
        num_signatures=len(boxes),
        boxes=boxes,
        message=f"Found {len(boxes)} signature(s)."
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Upload two signature images → get similarity score immediately.
    Score 0.0 = different, 1.0 = identical. is_match = True if score >= 0.75
    """
    if not file1.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file1 must be an image.")
    if not file2.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file2 must be an image.")

    bytes1 = await file1.read()
    bytes2 = await file2.read()

    score = compute_similarity(bytes1, bytes2)
    is_match = score >= 0.75

    return SimilarityResponse(
        similarity_score=score,
        is_match=is_match,
        message="Signatures match." if is_match else "Signatures do not match."
    )


# ── Async queue endpoints (job runs in background via worker) ─────────────────

@app.post("/queue/detect")
async def queue_detect(file: UploadFile = File(...)):
    """
    Same as /detect but runs in background via Celery worker.
    Returns a job_id — use /queue/result/{job_id} to get the result.
    """
    try:
        from worker.tasks import detect_task
    except ImportError:
        raise HTTPException(status_code=503, detail="Worker not available.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    job = detect_task.delay(image_bytes)
    return {"job_id": job.id, "status": "queued", "check_at": f"/queue/result/{job.id}"}


@app.post("/queue/similarity")
async def queue_similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Same as /similarity but runs in background via Celery worker.
    Returns a job_id — use /queue/result/{job_id} to get the result.
    """
    try:
        from worker.tasks import similarity_task
    except ImportError:
        raise HTTPException(status_code=503, detail="Worker not available.")

    bytes1 = await file1.read()
    bytes2 = await file2.read()
    job = similarity_task.delay(bytes1, bytes2)
    return {"job_id": job.id, "status": "queued", "check_at": f"/queue/result/{job.id}"}


@app.get("/queue/result/{job_id}")
def get_result(job_id: str):
    """
    Poll this endpoint with the job_id to get background job result.
    status: queued → started → done / failed
    """
    try:
        from worker.worker import celery_app
        job = celery_app.AsyncResult(job_id)
        if job.state == "PENDING":
            return {"job_id": job_id, "status": "queued"}
        elif job.state == "PROGRESS":
            return {"job_id": job_id, "status": "processing", "info": job.info}
        elif job.state == "SUCCESS":
            return {"job_id": job_id, "status": "done", "result": job.result}
        else:
            return {"job_id": job_id, "status": "failed", "error": str(job.info)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
