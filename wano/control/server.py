import contextlib
import json
import socket
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import ray
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from zeroconf import ServiceInfo, Zeroconf

from wano.control.db import Database
from wano.control.ray_manager import RayManager, get_local_ip
from wano.control.scheduler import Scheduler
from wano.execution.runner import execute_on_ray
from wano.models.compute import NodeCapabilities
from wano.models.job import JobStatus

app = FastAPI(title="Wano Control Plane")
db: Database | None = None
ray_manager: RayManager | None = None
scheduler: Scheduler | None = None
zeroconf_instance: Zeroconf | None = None
job_logs: dict[str, list[str]] = {}
_logs_lock = threading.Lock()
running_tasks: dict[str, list] = {}
_tasks_lock = threading.Lock()


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
    node_ids: list[str],
    compute: str,
    gpus: int | None,
    args: str | None = None,
    kwargs: str | None = None,
):
    try:
        result = execute_on_ray(job_id, function_code, node_ids, compute, gpus, args, kwargs)
        if db:
            result_json = json.dumps(result) if result is not None else None
            db.complete_job(job_id, result=result_json)
    except Exception as e:
        error_msg = str(e) + "\n" + traceback.format_exc()
        with _logs_lock:
            job_logs.setdefault(job_id, []).append(f"ERROR: {error_msg}")
        if db:
            db.complete_job(job_id, error=error_msg)


@app.post("/submit")
async def submit_job(
    background_tasks: BackgroundTasks,
    compute: str,
    gpus: int | None = None,
    function_code: str = "",
    args: str | None = None,
    kwargs: str | None = None,
):
    if not db or not scheduler:
        raise HTTPException(status_code=500, detail="Control plane not initialized")
    job_id = str(uuid.uuid4())
    job = db.create_job(job_id, compute, gpus, function_code, args, kwargs)
    available_compute = db.get_available_compute()
    node_ids = scheduler.schedule_job(job, available_compute)
    if not node_ids:
        return {"status": "pending", "job_id": job_id, "message": "No available compute"}
    db.assign_job(job_id, node_ids)
    background_tasks.add_task(
        _run_job, job_id, function_code, node_ids, compute, gpus, args, kwargs
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
        "jobs": [
            {
                "job_id": j.job_id,
                "compute": j.compute,
                "gpus": j.gpus,
                "status": j.status.value,
                "node_ids": j.node_ids,
                "result": j.result,
                "error": j.error,
            }
            for j in jobs
        ],
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "compute": job.compute,
        "gpus": job.gpus,
        "status": job.status.value,
        "node_ids": job.node_ids,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error": job.error,
        "result": job.result,
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail=f"Job is {job.status.value}, cannot cancel")
    with _tasks_lock:
        tasks = running_tasks.get(job_id, [])
        for task in tasks:
            ray.cancel(task, force=True)
        running_tasks.pop(job_id, None)
    _check_db().cancel_job(job_id)
    return {"status": "cancelled", "job_id": job_id}


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def generate():
        with _logs_lock:
            logs = job_logs.get(job_id, [])
        if not logs:
            yield "No logs available yet.\n"
        else:
            for line in logs:
                yield f"{line}\n"

    return StreamingResponse(generate(), media_type="text/plain")


def _retry_pending_jobs():
    if not db or not scheduler:
        return
    pending_jobs = db.get_pending_jobs()
    if not pending_jobs:
        return
    available_compute = db.get_available_compute()
    for job in pending_jobs:
        node_ids = scheduler.schedule_job(job, available_compute)
        if node_ids:
            db.assign_job(job.job_id, node_ids)
            threading.Thread(
                target=_run_job,
                args=(
                    job.job_id,
                    job.function_code or "",
                    node_ids,
                    job.compute,
                    job.gpus,
                    job.args,
                    job.kwargs,
                ),
                daemon=True,
            ).start()
            break


def _start_pending_job_retry():
    def retry_loop():
        while True:
            time.sleep(5)
            with contextlib.suppress(Exception):
                _retry_pending_jobs()

    threading.Thread(target=retry_loop, daemon=True).start()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
