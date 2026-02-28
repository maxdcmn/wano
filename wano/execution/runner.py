import base64
import contextlib
import json
import re
import tempfile
from pathlib import Path

import docker
import ray
import requests

from wano.control import log_store
from wano.execution.decorator import get_function_code

CONTAINER_IMAGE = "wano-executor:latest"


def submit_job(
    function_name: str,
    compute: str,
    gpus: int | None = None,
    control_plane_url: str = "http://localhost:8000",
    args: list | None = None,
    kwargs: dict | None = None,
    env_vars: dict | None = None,
    priority: int = 0,
    max_retries: int = 0,
    timeout_seconds: int | None = None,
    depends_on: list[str] | None = None,
    node_selector: dict[str, str] | None = None,
    namespace: str | None = None,
) -> str:
    function_code_bytes = get_function_code(function_name)
    if not function_code_bytes:
        raise ValueError(f"Function {function_name} not found in registry")
    payload: dict[str, object] = {
        "compute": compute,
        "gpus": gpus,
        "function_name": function_name,
        "function_code": base64.b64encode(function_code_bytes).decode("utf-8"),
        "priority": priority,
        "max_retries": max_retries,
    }
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds
    if depends_on:
        payload["depends_on"] = depends_on
    if node_selector:
        payload["node_selector"] = node_selector
    if namespace:
        payload["namespace"] = namespace
    if args is not None:
        payload["args"] = json.dumps(args)
    if kwargs is not None:
        payload["kwargs"] = json.dumps(kwargs)
    if env_vars is not None:
        payload["env_vars"] = json.dumps(env_vars)
    response = requests.post(
        f"{control_plane_url}/submit",
        json=payload,
    )
    response.raise_for_status()
    job_id = response.json().get("job_id")
    if isinstance(job_id, str):
        return job_id
    raise ValueError("Invalid response: missing job_id")


def _build_script(source_code: str, function_name: str) -> str:
    return f"""\
import json, os
{source_code}
_args = json.loads(os.environ.get("ARGS", "[]"))
_kwargs = json.loads(os.environ.get("KWARGS", "{{}}"))
for _k, _v in json.loads(os.environ.get("ENV_VARS", "{{}}")).items():
    os.environ[_k] = str(_v)
_result = {function_name}(*_args, **_kwargs)
with open("/tmp/result.json", "w") as f:
    json.dump(_result, f)
"""


def _run_container(
    job_id: str,
    source_code: str,
    function_name: str,
    args: list,
    kwargs: dict,
    env_vars: dict,
    compute: str,
) -> tuple:
    client = docker.from_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "job.py"
        script.write_text(_build_script(source_code, function_name))
        result_file = Path(tmpdir) / "result.json"
        result_file.touch()
        device_requests = None
        if compute == "gpu":
            device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        try:
            container = client.containers.run(
                CONTAINER_IMAGE,
                "python /job/job.py",
                volumes={
                    str(tmpdir): {"bind": "/job", "mode": "ro"},
                    str(result_file): {"bind": "/tmp/result.json", "mode": "rw"},
                },
                network_disabled=True,
                device_requests=device_requests,
                environment={
                    "ARGS": json.dumps(args),
                    "KWARGS": json.dumps(kwargs),
                    "ENV_VARS": json.dumps(env_vars),
                },
                detach=True,
            )
        except docker.errors.ContainerError as e:
            raise RuntimeError(e.stderr.decode() if e.stderr else str(e)) from e

        logs_buffer: list[str] = []
        try:
            for line in container.logs(stream=True):
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                if decoded:
                    log_store.append_lines(job_id, [decoded])
                    logs_buffer.append(decoded)
            status = container.wait()
            if status.get("StatusCode", 0) != 0:
                raise RuntimeError("\n".join(logs_buffer) or "Container exited with failure")
            result = json.loads(result_file.read_text()) if result_file.exists() else None
            return result, "\n".join(logs_buffer)
        finally:
            with contextlib.suppress(Exception):
                container.remove(force=True)


def execute_on_ray(
    job_id: str,
    function_code: str,
    node_ids: list,
    compute: str,
    gpus: int | None = None,
    args: str | None = None,
    kwargs: str | None = None,
    env_vars: str | None = None,
    function_name: str | None = None,
    ray_node_ids: list[str | None] | None = None,
    timeout_seconds: int | None = None,
):
    source_code = base64.b64decode(function_code).decode("utf-8")
    if not function_name:
        match = re.search(r"def\s+(\w+)\s*\(", source_code)
        if not match:
            raise ValueError("Could not find function definition in source code")
        function_name = match.group(1)

    parsed_args = json.loads(args) if args else []
    parsed_kwargs = json.loads(kwargs) if kwargs else {}
    parsed_env_vars = json.loads(env_vars) if env_vars else {}

    def run_task():
        return _run_container(
            job_id, source_code, function_name, parsed_args, parsed_kwargs, parsed_env_vars, compute
        )

    num_gpus = (gpus or 1) if compute == "gpu" else 0

    def _resolve_ray_node_id(wano_node_id: str) -> str | None:
        for node in ray.nodes():
            if not node.get("Alive", node.get("alive", True)):
                continue
            node_id = node.get("NodeID") or node.get("NodeId") or node.get("node_id")
            hostname = node.get("NodeManagerHostname") or node.get("Hostname")
            address = node.get("NodeManagerAddress")
            if wano_node_id in {node_id, hostname, address}:
                return node_id if isinstance(node_id, str) else None
        return None

    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except Exception:  # pragma: no cover - optional ray module
        NodeAffinitySchedulingStrategy = None

    def _make_task(ray_node_id: str | None, **resources):
        options = {}
        if ray_node_id and NodeAffinitySchedulingStrategy:
            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=ray_node_id, soft=False
            )
        return ray.remote(**resources)(run_task).options(**options).remote()

    def _resolve(wano_id: str, ray_id: str | None = None) -> str | None:
        return ray_id or _resolve_ray_node_id(wano_id)

    if compute == "gpu":
        assignments = list(node_ids[:num_gpus]) if node_ids else []
        if assignments and len(assignments) < num_gpus:
            assignments.extend([assignments[0]] * (num_gpus - len(assignments)))
        resolved = [
            _resolve(wid, rid) for wid, rid in zip(assignments, ray_node_ids or [], strict=False)
        ] + [_resolve_ray_node_id(wid) for wid in assignments[len(ray_node_ids or []) :]]
        tasks = [
            _make_task(resolved[i] if i < len(resolved) else None, num_gpus=1)
            for i in range(num_gpus)
        ]
    else:
        target = node_ids[0] if node_ids else None
        ray_target = _resolve(target, (ray_node_ids or [None])[0]) if target else None
        tasks = [_make_task(ray_target, num_cpus=1)]

    from wano.control.state import running_tasks, tasks_lock

    with tasks_lock:
        running_tasks[job_id] = tasks

    try:
        try:
            results = ray.get(tasks, timeout=timeout_seconds)
        except ray.exceptions.GetTimeoutError:
            for task in tasks:
                ray.cancel(task, force=True)
            raise TimeoutError(f"Job {job_id} timed out after {timeout_seconds} seconds") from None
        return [r for r, _ in results] if num_gpus > 1 else results[0][0]
    finally:
        with tasks_lock:
            running_tasks.pop(job_id, None)


def stream_logs(job_id: str, control_plane_url: str = "http://localhost:8000"):
    response = requests.get(f"{control_plane_url}/jobs/{job_id}/logs", stream=True, timeout=30)
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
