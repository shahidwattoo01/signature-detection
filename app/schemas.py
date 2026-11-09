from pydantic import BaseModel
from typing import List


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class DetectionResponse(BaseModel):
    num_signatures: int
    boxes: List[BoundingBox]
    message: str


class SimilarityResponse(BaseModel):
    similarity_score: float
    is_match: bool
    message: str
