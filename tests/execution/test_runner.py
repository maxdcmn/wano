import base64
import json

import docker
import pytest

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
def test_execute_on_ray_cpu():
    function_code = base64.b64encode(b"def task(): return 123").decode()
    result = execute_on_ray("test-job", function_code, ["node1"], "cpu")
    assert result == 123


@requires_docker
def test_execute_on_ray_with_args_and_kwargs():
    function_code = base64.b64encode(b"def task(x, y, z=1): return x + y + z").decode()
    args = json.dumps([10, 20])
    kwargs = json.dumps({"z": 5})
    result = execute_on_ray(
        "test-job-both", function_code, ["node1"], "cpu", args=args, kwargs=kwargs
    )
    assert result == 35


@requires_docker
def test_execute_on_ray_handles_errors(ray_cluster):
    function_code = base64.b64encode(b"def task(): raise ValueError('test error')").decode()
    with pytest.raises(RuntimeError):
        execute_on_ray("test-job", function_code, ["node1"], "cpu")


@requires_docker
def test_execute_on_ray_with_env_vars(ray_cluster):
    function_code = base64.b64encode(
        b"import os\ndef task(): return os.environ.get('TEST_VAR', 'not_set')"
    ).decode()
    env_vars = json.dumps({"TEST_VAR": "test_value"})
    result = execute_on_ray("test-job-env", function_code, ["node1"], "cpu", env_vars=env_vars)
    assert result == "test_value"


@pytest.mark.skip(reason="Requires GPU")
def test_execute_on_ray_gpu():
    function_code = base64.b64encode(b"def task(): return 456").decode()
    execute_on_ray("test-job", function_code, ["node1"], "gpu", 1)
