from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from wano.models.job import Job, JobStatus


class HasGetJob(Protocol):
    def get_job(self, job_id: str) -> Job | None: ...


_LOG_DIR = Path.home() / ".wano" / "logs"
_LOCK = threading.Lock()


def _ensure_dir() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def get_log_path(job_id: str) -> Path:
    return _ensure_dir() / f"{job_id}.log"


def append_lines(job_id: str, lines: Iterable[str]):
    items = [line.rstrip("\n") for line in lines if line is not None]
    if not items:
        return
    path = get_log_path(job_id)
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        for line in items:
            f.write(f"{line}\n")


def read_lines(job_id: str) -> list[str]:
    path = get_log_path(job_id)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _read_new(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], offset
    with open(path, encoding="utf-8") as f:
        f.seek(offset)
        chunk = f.read()
        new_offset = f.tell()
    if not chunk:
        return [], new_offset
    return chunk.splitlines(), new_offset


def stream_logs(job_id: str, get_db: Callable[[], HasGetJob], poll_interval: float = 0.5):
    path = get_log_path(job_id)
    offset = 0
    sent_any = False
    while True:
        lines, offset = _read_new(path, offset)
        if lines:
            sent_any = True
            for line in lines:
                yield f"{line}\n"
        job = get_db().get_job(job_id)
        if job and job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            lines, offset = _read_new(path, offset)
            for line in lines:
                yield f"{line}\n"
            if not sent_any and not lines:
                yield "No logs available.\n"
            break
        if not lines:
            time.sleep(poll_interval)
