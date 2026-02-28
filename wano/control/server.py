import contextlib
import json
import socket
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ray
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from zeroconf import ServiceInfo, Zeroconf

from wano.control import log_store
from wano.control.db import Database
from wano.control.ray_manager import RayManager, get_local_ip
from wano.control.scheduler import Scheduler
from wano.control.state import running_tasks, tasks_lock
from wano.execution.runner import execute_on_ray
from wano.models.compute import NodeCapabilities
from wano.models.job import JobStatus
from wano.models.quota import ResourceQuota


class SubmitRequest(BaseModel):
    compute: str
    gpus: int | None = None
    priority: int = 0
    max_retries: int = 0
    timeout_seconds: int | None = None
    function_name: str | None = None
    function_code: str = ""
    args: str | None = None
    kwargs: str | None = None
    env_vars: str | None = None
    depends_on: list[str] | None = None
    node_selector: dict[str, str] | None = None
    namespace: str | None = None


class QuotaRequest(BaseModel):
    namespace: str
    max_cpu_jobs: int | None = None
    max_gpu_jobs: int | None = None


app = FastAPI(title="Wano Control Plane")
db: Database | None = None
ray_manager: RayManager | None = None
scheduler: Scheduler | None = None
zeroconf_instance: Zeroconf | None = None


def init_control_plane(db_path: Path, ray_port: int = 10001, api_port: int = 8000):
    global db, ray_manager, scheduler
    db = Database(db_path)
    ray_manager = RayManager()
    ray_manager.start(ray_port)
    scheduler = Scheduler()
    start_mdns_advertising(api_port)
    _start_pending_job_retry()


def start_mdns_advertising(port: int = 8000):
    global zeroconf_instance
    local_ip = get_local_ip()
    info = ServiceInfo(
        "_wano._tcp.local.",
        "wano-control-plane._wano._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=port,
    )
    zeroconf_instance = Zeroconf()
    zeroconf_instance.register_service(info)


def _check_db() -> Database:
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


@app.post("/register")
async def register_node(capabilities: dict[str, Any]):
    db_instance = _check_db()
    node_caps = NodeCapabilities.from_dict(capabilities)
    db_instance.register_node(node_caps.node_id, node_caps)
    return {"status": "registered"}


@app.post("/heartbeat")
async def heartbeat(capabilities: dict[str, Any]):
    db_instance = _check_db()
    node_caps = NodeCapabilities.from_dict(capabilities)
    db_instance.update_heartbeat(node_caps.node_id)
    db_instance.register_node(node_caps.node_id, node_caps)
    return {"status": "ok"}


def _run_job(
    job_id: str,
    function_code: str,
    function_name: str | None,
    node_ids: list[str],
    ray_node_ids: list[str | None] | None,
    compute: str,
    gpus: int | None,
    args: str | None = None,
    kwargs: str | None = None,
    env_vars: str | None = None,
    timeout_seconds: int | None = None,
):
    try:
        result = execute_on_ray(
            job_id,
            function_code,
            node_ids,
            compute,
            gpus,
            args,
            kwargs,
            env_vars,
            function_name=function_name,
            ray_node_ids=ray_node_ids,
            timeout_seconds=timeout_seconds,
        )
        if db:
            result_json = json.dumps(result) if result is not None else None
            db.complete_job(job_id, result=result_json)
    except TimeoutError as e:
        error_msg = str(e)
        log_store.append_lines(job_id, [f"TIMEOUT: {error_msg}"])
        if db:
            db.fail_job_timeout(job_id, error=error_msg)
            db.cascade_failure(job_id)
    except Exception as e:
        error_msg = str(e) + "\n" + traceback.format_exc()
        log_store.append_lines(job_id, [f"ERROR: {error_msg}"])
        if db:
            db.complete_job(job_id, error=error_msg)
            job = db.get_job(job_id)
            if job and job.status == JobStatus.FAILED:
                db.cascade_failure(job_id)


@app.post("/submit")
async def submit_job(body: SubmitRequest, background_tasks: BackgroundTasks):
    if not db or not scheduler:
        raise HTTPException(status_code=500, detail="Control plane not initialized")
    if body.depends_on:
        missing = db.validate_depends_on(body.depends_on)
        if missing:
            raise HTTPException(status_code=400, detail=f"Unknown dependency job IDs: {missing}")
    job_id = str(uuid.uuid4())
    job = db.create_job(
        job_id,
        body.compute,
        body.gpus,
        body.function_name,
        body.function_code,
        body.args,
        body.kwargs,
        body.env_vars,
        priority=body.priority,
        max_retries=body.max_retries,
        timeout_seconds=body.timeout_seconds,
        depends_on=body.depends_on,
        node_selector=body.node_selector,
        namespace=body.namespace,
    )
    if body.depends_on and not db.deps_satisfied(body.depends_on):
        return {"status": "pending", "job_id": job_id, "message": "Waiting on dependencies"}
    available_compute = db.get_available_compute()
    node_usage = db.get_node_usage()
    node_labels = db.get_node_labels()
    quota = db.get_quota(body.namespace) if body.namespace else None
    ns_usage = db.get_namespace_usage(body.namespace) if body.namespace else None
    node_ids = scheduler.schedule_job(
        job, available_compute, node_usage, node_labels, quota=quota, namespace_usage=ns_usage
    )
    if not node_ids:
        return {"status": "pending", "job_id": job_id, "message": "No available compute"}
    db.assign_job(job_id, node_ids)
    ray_node_ids = db.get_ray_node_ids(node_ids)
    background_tasks.add_task(
        _run_job,
        job_id,
        body.function_code,
        body.function_name,
        node_ids,
        ray_node_ids,
        body.compute,
        body.gpus,
        body.args,
        body.kwargs,
        body.env_vars,
        body.timeout_seconds,
    )
    return {"status": "submitted", "job_id": job_id, "node_ids": node_ids}


@app.get("/compute")
async def get_compute():
    return {"compute": _check_db().get_available_compute()}


@app.get("/ray-address")
async def get_ray_address():
    if not ray_manager:
        raise HTTPException(status_code=500, detail="Ray manager not initialized")
    address = ray_manager.get_address()
    if not address:
        raise HTTPException(status_code=500, detail="Ray cluster not running")
    return {"ray_address": address}


@app.get("/status")
async def get_status():
    db_instance = _check_db()
    jobs = db_instance.get_all_jobs()
    return {
        "compute": db_instance.get_available_compute(),
        "active_nodes": db_instance.get_active_nodes(),
        "nodes": db_instance.get_nodes(),
        "jobs": [j.to_dict() for j in jobs],
    }


@app.post("/nodes/{node_id}/cordon")
async def cordon_node(node_id: str):
    if not _check_db().set_node_status(node_id, "cordoned"):
        raise HTTPException(status_code=404, detail="Node not found")
    return {"status": "cordoned", "node_id": node_id}


@app.post("/nodes/{node_id}/uncordon")
async def uncordon_node(node_id: str):
    if not _check_db().set_node_status(node_id, "active"):
        raise HTTPException(status_code=404, detail="Node not found")
    return {"status": "active", "node_id": node_id}


@app.post("/quotas")
async def set_quota(body: QuotaRequest):
    db_instance = _check_db()
    db_instance.create_or_update_quota(body.namespace, body.max_cpu_jobs, body.max_gpu_jobs)
    return {"status": "ok", "namespace": body.namespace}


@app.get("/quotas")
async def list_quotas():
    db_instance = _check_db()
    quotas = db_instance.get_all_quotas()
    return {
        "quotas": [
            {
                "namespace": q.namespace,
                "max_cpu_jobs": q.max_cpu_jobs,
                "max_gpu_jobs": q.max_gpu_jobs,
            }
            for q in quotas
        ]
    }


@app.get("/quotas/{namespace}")
async def get_quota_detail(namespace: str):
    db_instance = _check_db()
    quota = db_instance.get_quota(namespace)
    if not quota:
        raise HTTPException(status_code=404, detail="Quota not found")
    usage = db_instance.get_namespace_usage(namespace)
    return {
        "namespace": quota.namespace,
        "max_cpu_jobs": quota.max_cpu_jobs,
        "max_gpu_jobs": quota.max_gpu_jobs,
        "usage": usage,
    }


@app.delete("/quotas/{namespace}")
async def delete_quota(namespace: str):
    if not _check_db().delete_quota(namespace):
        raise HTTPException(status_code=404, detail="Quota not found")
    return {"status": "deleted", "namespace": namespace}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.RUNNING, JobStatus.PENDING):
        raise HTTPException(status_code=400, detail=f"Job is {job.status.value}, cannot cancel")
    if job.status == JobStatus.RUNNING:
        with tasks_lock:
            tasks = running_tasks.get(job_id, [])
            for task in tasks:
                ray.cancel(task, force=True)
            running_tasks.pop(job_id, None)
    db_instance = _check_db()
    db_instance.cancel_job(job_id)
    db_instance.cascade_failure(job_id)
    return {"status": "cancelled", "job_id": job_id}


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def generate():
        yield from log_store.stream_logs(job_id, _check_db)

    return StreamingResponse(generate(), media_type="text/plain")


def _retry_pending_jobs():
    if not db or not scheduler:
        return
    pending_jobs = db.get_pending_jobs()
    if not pending_jobs:
        return
    available_compute = db.get_available_compute()
    node_usage = db.get_node_usage()
    node_labels = db.get_node_labels()
    quota_cache: dict[str, ResourceQuota | None] = {}
    usage_cache: dict[str, dict[str, int]] = {}
    for job in pending_jobs:
        with tasks_lock:
            if job.job_id in running_tasks:
                continue
        ns = job.namespace
        if ns and ns not in quota_cache:
            quota_cache[ns] = db.get_quota(ns)
            usage_cache[ns] = db.get_namespace_usage(ns)
        quota = quota_cache.get(ns) if ns else None
        ns_usage = usage_cache.get(ns) if ns else None
        node_ids = scheduler.schedule_job(
            job, available_compute, node_usage, node_labels, quota=quota, namespace_usage=ns_usage
        )
        if node_ids:
            db.assign_job(job.job_id, node_ids)
            if ns and ns in usage_cache:
                key = "gpu" if job.compute == "gpu" else "cpu"
                usage_cache[ns][key] = usage_cache[ns].get(key, 0) + 1
            ray_node_ids = db.get_ray_node_ids(node_ids)
            threading.Thread(
                target=_run_job,
                args=(
                    job.job_id,
                    job.function_code or "",
                    job.function_name,
                    node_ids,
                    ray_node_ids,
                    job.compute,
                    job.gpus,
                    job.args,
                    job.kwargs,
                    job.env_vars,
                    job.timeout_seconds,
                ),
                daemon=True,
            ).start()


def _check_timed_out_jobs():
    if not db:
        return
    for job in db.get_running_jobs():
        if job.timeout_seconds is None or job.started_at is None:
            continue
        elapsed = (datetime.now(UTC) - job.started_at).total_seconds()
        if elapsed > job.timeout_seconds:
            with tasks_lock:
                tasks = running_tasks.get(job.job_id, [])
                for task in tasks:
                    ray.cancel(task, force=True)
                running_tasks.pop(job.job_id, None)
            error_msg = f"Job {job.job_id} timed out after {job.timeout_seconds} seconds"
            log_store.append_lines(job.job_id, [f"TIMEOUT: {error_msg}"])
            db.fail_job_timeout(job.job_id, error=error_msg)
            db.cascade_failure(job.job_id)


def _start_pending_job_retry():
    def retry_loop():
        while True:
            time.sleep(5)
            with contextlib.suppress(Exception):
                _retry_pending_jobs()
            with contextlib.suppress(Exception):
                _check_timed_out_jobs()

    threading.Thread(target=retry_loop, daemon=True).start()


def shutdown():
    if zeroconf_instance:
        with contextlib.suppress(Exception):
            zeroconf_instance.close()
    if db:
        with contextlib.suppress(Exception):
            db.close()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        shutdown()
