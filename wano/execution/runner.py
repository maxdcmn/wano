import base64
import io
import json
import re
import sys
from collections.abc import Callable

import ray
import requests

from wano.execution.decorator import get_function_code


def submit_job(
    function_name: str,
    compute: str,
    gpus: int | None = None,
    control_plane_url: str = "http://localhost:8000",
) -> str:
    function_code_bytes = get_function_code(function_name)
    if not function_code_bytes:
        raise ValueError(f"Function {function_name} not found in registry")
    response = requests.post(
        f"{control_plane_url}/submit",
        json={
            "compute": compute,
            "gpus": gpus,
            "function_code": base64.b64encode(function_code_bytes).decode("utf-8"),
        },
    )
    response.raise_for_status()
    job_id = response.json().get("job_id")
    if isinstance(job_id, str):
        return job_id
    raise ValueError("Invalid response: missing job_id")


def _capture_output(func: Callable) -> Callable:
    def wrapper():
        log_buffer = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_buffer
        sys.stderr = log_buffer
        try:
            result = func()
            return result, log_buffer.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return wrapper


def _store_logs(job_id: str, logs: str):
    if logs:
        from wano.control.server import _logs_lock, job_logs

        with _logs_lock:
            job_logs.setdefault(job_id, []).extend(logs.splitlines())


def execute_on_ray(
    job_id: str,
    function_code: str,
    node_ids: list,
    compute: str,
    gpus: int | None = None,
    args: str | None = None,
    kwargs: str | None = None,
):
    source_code = base64.b64decode(function_code).decode("utf-8")
    namespace: dict[str, Callable] = {}
    exec(compile(source_code, "<string>", "exec"), namespace)

    match = re.search(r"def\s+(\w+)\s*\(", source_code)
    if not match:
        raise ValueError("Could not find function definition in source code")
    func_name = match.group(1)

    if func_name not in namespace:
        raise ValueError(f"Function {func_name} not found in executed namespace")
    func = namespace[func_name]

    parsed_args = json.loads(args) if args else []
    parsed_kwargs = json.loads(kwargs) if kwargs else {}

    def call_func():
        return func(*parsed_args, **parsed_kwargs)

    wrapped_func = _capture_output(call_func)
    num_gpus = gpus or 1 if compute == "gpu" else 0

    if num_gpus > 1:
        bundles = [{"GPU": 1} for _ in range(num_gpus)]
        pg = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        tasks = [
            ray.remote(num_gpus=1)(lambda: wrapped_func()).options(placement_group=pg).remote()
            for _ in range(num_gpus)
        ]
    elif num_gpus == 1:
        tasks = [ray.remote(num_gpus=1)(lambda: wrapped_func()).remote()]
    else:
        tasks = [ray.remote(num_cpus=1)(lambda: wrapped_func()).remote()]

    from wano.control.server import _tasks_lock, running_tasks

    with _tasks_lock:
        running_tasks[job_id] = tasks

    try:
        results = ray.get(tasks)
        for _result, logs in results:
            _store_logs(job_id, logs)
        return [r for r, _ in results] if num_gpus > 1 else results[0][0]
    finally:
        with _tasks_lock:
            running_tasks.pop(job_id, None)


def stream_logs(job_id: str, control_plane_url: str = "http://localhost:8000"):
    for line in requests.get(f"{control_plane_url}/jobs/{job_id}/logs", stream=True).iter_lines():
        if line:
            print(line.decode("utf-8"))
