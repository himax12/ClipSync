"""Pydantic models for API requests/responses"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "complete", "error", "queued"
    progress: int  # 0-100
    message: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    stats: Optional[Dict] = None


class TimelineEntry(BaseModel):
    aroll_start: float
    aroll_end: float
    aroll_text: str
    broll_clip: Optional[str]
    broll_name: Optional[str]
    similarity: float
    duration: float


class ProcessingResult(BaseModel):
    job_id: str
    timeline: List[TimelineEntry]
    stats: Dict
    output_video: str


class IndexStats(BaseModel):
    total_vectors: int
    dimension: int
    on_gpu: bool
    num_clips: int
