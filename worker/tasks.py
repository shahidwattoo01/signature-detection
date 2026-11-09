from worker.worker import celery_app
from app.detector import detect_signatures
from app.similarity import compute_similarity


@celery_app.task(bind=True, name="tasks.detect")
def detect_task(self, image_bytes: bytes):
    """
    Background task: runs signature detection.
    Called via queue so API does not block.
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "detecting..."})
        results = detect_signatures(image_bytes)
        return {
            "status": "done",
            "num_signatures": len(results),
            "boxes": results
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery_app.task(bind=True, name="tasks.similarity")
def similarity_task(self, image1_bytes: bytes, image2_bytes: bytes):
    """
    Background task: compares two signatures.
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "comparing..."})
        score = compute_similarity(image1_bytes, image2_bytes)
        return {
            "status": "done",
            "similarity_score": score,
            "is_match": score >= 0.75
        }
    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
