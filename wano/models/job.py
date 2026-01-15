from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    compute: str
    gpus: int | None = None
    status: JobStatus = JobStatus.PENDING
    node_ids: list[str] | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    function_code: str | None = None
    error: str | None = None
    result: str | None = None
    args: str | None = None
    kwargs: str | None = None
