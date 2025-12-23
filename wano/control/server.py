import contextlib
import socket
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from zeroconf import ServiceInfo, Zeroconf

from wano.control.db import Database
from wano.control.ray_manager import RayManager
from wano.control.scheduler import Scheduler
from wano.models.compute import NodeCapabilities

app = FastAPI(title="Wano Control Plane")
db: Database | None = None
ray_manager: RayManager | None = None
scheduler: Scheduler | None = None
zeroconf_instance: Zeroconf | None = None
join_token: str = "wano-default-token"


def init_control_plane(db_path: Path, ray_port: int = 10001):
    global db, ray_manager, scheduler
    db = Database(db_path)
    ray_manager = RayManager()
    ray_manager.start(ray_port)
    scheduler = Scheduler()
    start_mdns_advertising()


def start_mdns_advertising():
    global zeroconf_instance
    local_ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        with contextlib.suppress(Exception):
            local_ip = socket.gethostbyname(socket.gethostname())
    if not local_ip:
        local_ip = "127.0.0.1"
    info = ServiceInfo(
        "_wano._tcp.local.",
        "wano-control-plane._wano._tcp.local.",
        addresses=[socket.inet_aton(local_ip)],
        port=8000,
        properties={"token": join_token.encode()},
    )
    zeroconf_instance = Zeroconf()
    zeroconf_instance.register_service(info)


def _check_db():
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")


@app.post("/register")
async def register_node(capabilities: dict):
    _check_db()
    assert db is not None
    node_caps = NodeCapabilities.from_dict(capabilities)
    db.register_node(node_caps.node_id, node_caps)
    return {"status": "registered", "token": join_token}


@app.post("/heartbeat")
async def heartbeat(data: dict):
    _check_db()
    assert db is not None
    node_id = data.get("node_id")
    if not node_id:
        raise HTTPException(status_code=400, detail="node_id required")
    db.update_heartbeat(node_id)
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
    _check_db()
    assert db is not None
    return {"compute": db.get_available_compute()}


@app.get("/status")
async def get_status():
    _check_db()
    assert db is not None
    jobs = db.get_all_jobs()
    return {
        "compute": db.get_available_compute(),
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
    _check_db()
    assert db is not None
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        (f"Logs for job {job_id}\n", "Not implemented yet\n"), media_type="text/plain"
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
