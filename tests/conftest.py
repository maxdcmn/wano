import os
import tempfile
import warnings
from pathlib import Path

import pytest
import ray

from wano.control.db import Database


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = Database(db_path)
    yield db
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
