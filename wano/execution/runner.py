import base64

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


def execute_on_ray(
    job_id: str, function_code: bytes, node_ids: list, compute: str, gpus: int | None = None
):
    import pickle

    func = pickle.loads(function_code)
    num_gpus = gpus or 1 if compute == "gpu" else 0
    if num_gpus > 1:
        bundles = [{"GPU": 1} for _ in range(num_gpus)]
        pg = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        ray.get(
            [
                ray.remote(num_gpus=1)(lambda: func()).options(placement_group=pg).remote()
                for _ in range(num_gpus)
            ]
        )
    elif num_gpus == 1:
        ray.get(ray.remote(num_gpus=1)(lambda: func()).remote())
    else:
        ray.get(ray.remote(num_cpus=1)(lambda: func()).remote())


def stream_logs(job_id: str, control_plane_url: str = "http://localhost:8000"):
    for line in requests.get(f"{control_plane_url}/jobs/{job_id}/logs", stream=True).iter_lines():
        if line:
            print(line.decode("utf-8"))
