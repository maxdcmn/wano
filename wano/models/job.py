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
    priority: int = 0
    max_retries: int = 0
    attempts: int = 0
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    function_name: str | None = None
    function_code: str | None = None
    error: str | None = None
    result: str | None = None
    args: str | None = None
    kwargs: str | None = None
    env_vars: str | None = None
    timeout_seconds: int | None = None
    depends_on: list[str] | None = None
    node_selector: dict[str, str] | None = None
    namespace: str | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "compute": self.compute,
            "gpus": self.gpus,
            "status": self.status.value,
            "node_ids": self.node_ids,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "attempts": self.attempts,
            "function_name": self.function_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result": self.result,
            "timeout_seconds": self.timeout_seconds,
            "depends_on": self.depends_on,
            "node_selector": self.node_selector,
            "namespace": self.namespace,
        }
