import socket
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from zeroconf import ServiceInfo, Zeroconf

from wano.control.db import Database
from wano.control.ray_manager import RayManager, get_local_ip
from wano.control.scheduler import Scheduler
from wano.models.compute import NodeCapabilities

app = FastAPI(title="Wano Control Plane")
db: Database | None = None
ray_manager: RayManager | None = None
scheduler: Scheduler | None = None
zeroconf_instance: Zeroconf | None = None
join_token: str = "wano-default-token"


def init_control_plane(db_path: Path, ray_port: int = 10001, api_port: int = 8000):
    global db, ray_manager, scheduler
    db = Database(db_path)
    ray_manager = RayManager()
    ray_manager.start(ray_port)
    scheduler = Scheduler()
    start_mdns_advertising(api_port)


def start_mdns_advertising(port: int = 8000):
    global zeroconf_instance
    local_ip = get_local_ip()
    info = ServiceInfo(
        "_wano._tcp.local.",
        "wano-control-plane._wano._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties={"token": join_token.encode()},
    )
    zeroconf_instance = Zeroconf()
    zeroconf_instance.register_service(info)


def _check_db() -> Database:
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


@app.post("/register")
async def register_node(capabilities: dict):
    db_instance = _check_db()
    node_caps = NodeCapabilities.from_dict(capabilities)
    db_instance.register_node(node_caps.node_id, node_caps)
    return {"status": "registered", "token": join_token}


@app.post("/heartbeat")
async def heartbeat(capabilities: dict):
    db_instance = _check_db()
    node_caps = NodeCapabilities.from_dict(capabilities)
    db_instance.update_heartbeat(node_caps.node_id)
    db_instance.register_node(node_caps.node_id, node_caps)
    return {"status": "ok"}


@app.post("/submit")
async def submit_job(compute: str, gpus: int | None = None, function_code: str = ""):
    if not db or not scheduler:
        raise HTTPException(status_code=500, detail="Control plane not initialized")
    job_id = str(uuid.uuid4())
    job = db.create_job(job_id, compute, gpus, function_code)
    available_compute = db.get_available_compute()
    node_ids = scheduler.schedule_job(job, available_compute)
    if not node_ids:
        return {"status": "pending", "job_id": job_id, "message": "No available compute"}
    db.assign_job(job_id, node_ids)
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
            }
            for j in jobs
        ],
    }


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    job = _check_db().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        (f"Logs for job {job_id}\n", "Not implemented yet\n"), media_type="text/plain"
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
