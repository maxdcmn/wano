import base64

import pytest

from wano.execution.runner import execute_on_ray


def test_execute_on_ray_cpu():
    function_code = base64.b64encode(b"def task(): return 123").decode()
    execute_on_ray("test-job", function_code, ["node1"], "cpu", None)


def test_execute_on_ray_handles_errors():
    function_code = base64.b64encode(b"def task(): raise ValueError('test error')").decode()
    with pytest.raises(ValueError, match="test error"):
        execute_on_ray("test-job", function_code, ["node1"], "cpu", None)


@pytest.mark.skip(reason="Requires GPU")
def test_execute_on_ray_gpu():
    function_code = base64.b64encode(b"def task(): return 456").decode()
    execute_on_ray("test-job", function_code, ["node1"], "gpu", 1)
