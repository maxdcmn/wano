import sqlite3
from datetime import UTC, datetime, timedelta

from conftest import make_job

import wano.control.server as server_mod
from wano.control.server import _check_timed_out_jobs
from wano.models.job import JobStatus


def test_check_timed_out_jobs_cancels_expired(db):
    db.create_job(make_job("job-expired", timeout_seconds=10))
    db.assign_job("job-expired", ["node1"])

    past = (datetime.now(UTC) - timedelta(seconds=20)).isoformat()
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE jobs SET started_at = ? WHERE job_id = ?", (past, "job-expired"))
        conn.commit()

    original_db = server_mod.db
    server_mod.db = db
    try:
        _check_timed_out_jobs()
    finally:
        server_mod.db = original_db

    job = db.get_job("job-expired")
    assert job.status == JobStatus.FAILED
    assert "timed out" in job.error


def test_check_timed_out_jobs_ignores_no_timeout(db):
    db.create_job(make_job("job-no-to"))
    db.assign_job("job-no-to", ["node1"])

    original_db = server_mod.db
    server_mod.db = db
    try:
        _check_timed_out_jobs()
    finally:
        server_mod.db = original_db

    job = db.get_job("job-no-to")
    assert job.status == JobStatus.RUNNING
