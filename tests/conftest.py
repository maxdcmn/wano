import os
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest
import ray

from wano.control.db import Database
from wano.models.job import Job, JobStatus


def make_job(
    job_id, compute="cpu", gpus=None, function_name=None, function_code="def f(): pass", **kwargs
):
    return Job(
        job_id=job_id,
        compute=compute,
        gpus=gpus,
        function_name=function_name,
        function_code=function_code,
        status=JobStatus.PENDING,
        created_at=datetime.now(UTC),
        **kwargs,
    )


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = Database(db_path)
    yield db
    db.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def ray_cluster():
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    if ray.is_initialized():
        ray.shutdown()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*RAY_ACCEL.*")
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()
