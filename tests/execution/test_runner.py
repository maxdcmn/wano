import base64
import json

import pytest

from wano.execution.runner import execute_on_ray


def test_execute_on_ray_cpu():
    function_code = base64.b64encode(b"def task(): return 123").decode()
    result = execute_on_ray("test-job", function_code, ["node1"], "cpu", None)
    assert result == 123


def test_execute_on_ray_with_args_and_kwargs():
    function_code = base64.b64encode(b"def task(x, y, z=1): return x + y + z").decode()
    args = json.dumps([10, 20])
    kwargs = json.dumps({"z": 5})
    result = execute_on_ray(
        "test-job-both", function_code, ["node1"], "cpu", None, args=args, kwargs=kwargs
    )
    assert result == 35


def test_execute_on_ray_handles_errors():
    function_code = base64.b64encode(b"def task(): raise ValueError('test error')").decode()
    with pytest.raises(ValueError, match="test error"):
        execute_on_ray("test-job", function_code, ["node1"], "cpu", None)


@pytest.mark.skip(reason="Requires GPU")
def test_execute_on_ray_gpu():
    function_code = base64.b64encode(b"def task(): return 456").decode()
    execute_on_ray("test-job", function_code, ["node1"], "gpu", 1)
