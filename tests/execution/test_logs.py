import base64

import docker
import pytest

from wano.control.server import job_logs
from wano.execution.runner import CONTAINER_IMAGE, execute_on_ray


def _docker_available():
    try:
        client = docker.from_env()
        client.images.get(CONTAINER_IMAGE)
        return True
    except Exception:
        return False


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason=f"Docker not available or {CONTAINER_IMAGE} not built"
)


@requires_docker
def test_captures_stdout_logs():
    function_code = base64.b64encode(b"def task(): print('test output'); return 42").decode()
    execute_on_ray("test-job-logs", function_code, ["node1"], "cpu")

    assert "test-job-logs" in job_logs
    assert any("test output" in line for line in job_logs["test-job-logs"])


@requires_docker
def test_captures_multiple_lines():
    function_code = base64.b64encode(
        b"def task(): print('line1'); print('line2'); return 0"
    ).decode()
    execute_on_ray("test-job-multi", function_code, ["node1"], "cpu")

    logs = job_logs.get("test-job-multi", [])
    assert len(logs) >= 2
    assert any("line1" in line for line in logs)
    assert any("line2" in line for line in logs)
