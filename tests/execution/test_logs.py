import base64

from wano.control.server import job_logs
from wano.execution.runner import execute_on_ray


def test_captures_stdout_logs():
    function_code = base64.b64encode(b"def task(): print('test output'); return 42").decode()
    execute_on_ray("test-job-logs", function_code, ["node1"], "cpu", None)

    assert "test-job-logs" in job_logs
    assert any("test output" in line for line in job_logs["test-job-logs"])


def test_captures_multiple_lines():
    function_code = base64.b64encode(
        b"def task(): print('line1'); print('line2'); return 0"
    ).decode()
    execute_on_ray("test-job-multi", function_code, ["node1"], "cpu", None)

    logs = job_logs.get("test-job-multi", [])
    assert len(logs) >= 2
    assert any("line1" in line for line in logs)
    assert any("line2" in line for line in logs)
