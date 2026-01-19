import base64
import json
import re
import tempfile
from pathlib import Path

import docker
import ray
import requests

from wano.execution.decorator import get_function_code

CONTAINER_IMAGE = "wano-executor:latest"


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


def _build_script(source_code: str, func_name: str) -> str:
    return f"""\
import json, os
{source_code}
_args = json.loads(os.environ.get("ARGS", "[]"))
_kwargs = json.loads(os.environ.get("KWARGS", "{{}}"))
for _k, _v in json.loads(os.environ.get("ENV_VARS", "{{}}")).items():
    os.environ[_k] = str(_v)
_result = {func_name}(*_args, **_kwargs)
with open("/tmp/result.json", "w") as f:
    json.dump(_result, f)
"""


def _run_container(
    source_code: str,
    func_name: str,
    args: list,
    kwargs: dict,
    env_vars: dict,
    compute: str,
) -> tuple:
    client = docker.from_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "job.py"
        script.write_text(_build_script(source_code, func_name))
        result_file = Path(tmpdir) / "result.json"
        result_file.touch()
        device_requests = None
        if compute == "gpu":
            device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        try:
            output = client.containers.run(
                CONTAINER_IMAGE,
                "python /job/job.py",
                volumes={
                    str(tmpdir): {"bind": "/job", "mode": "ro"},
                    str(result_file): {"bind": "/tmp/result.json", "mode": "rw"},
                },
                remove=True,
                network_disabled=True,
                device_requests=device_requests,
                environment={
                    "ARGS": json.dumps(args),
                    "KWARGS": json.dumps(kwargs),
                    "ENV_VARS": json.dumps(env_vars),
                },
            )
        except docker.errors.ContainerError as e:
            raise RuntimeError(e.stderr.decode() if e.stderr else str(e)) from e

        logs = output.decode()
        result = json.loads(result_file.read_text()) if result_file.exists() else None
        return result, logs


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
    env_vars: str | None = None,
):
    source_code = base64.b64decode(function_code).decode("utf-8")
    match = re.search(r"def\s+(\w+)\s*\(", source_code)
    if not match:
        raise ValueError("Could not find function definition in source code")
    func_name = match.group(1)

    parsed_args = json.loads(args) if args else []
    parsed_kwargs = json.loads(kwargs) if kwargs else {}
    parsed_env_vars = json.loads(env_vars) if env_vars else {}

    def run_task():
        return _run_container(
            source_code, func_name, parsed_args, parsed_kwargs, parsed_env_vars, compute
        )

    num_gpus = gpus or 1 if compute == "gpu" else 0

    if num_gpus > 1:
        bundles = [{"GPU": 1} for _ in range(num_gpus)]
        pg = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        tasks = [
            ray.remote(num_gpus=1)(run_task).options(placement_group=pg).remote()
            for _ in range(num_gpus)
        ]
    elif num_gpus == 1:
        tasks = [ray.remote(num_gpus=1)(run_task).remote()]
    else:
        tasks = [ray.remote(num_cpus=1)(run_task).remote()]

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
