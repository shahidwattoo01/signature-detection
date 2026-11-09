from celery import Celery

# Redis is the message queue (broker) and result storage (backend)
celery_app = Celery(
    "signature_worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",
    include=["worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,  # results kept for 1 hour
)
