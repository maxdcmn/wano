import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import click
import requests

from wano.agent.worker import NodeAgent
from wano.control.process_manager import (
    get_pid,
    get_pid_file,
    is_process_running,
    kill_process,
    save_pid,
    start_detached,
)
from wano.execution.runner import stream_logs


@click.group()
def cli():
    pass


@cli.command()
@click.option("--port", default=8000, help="API server port")
@click.option("--ray-port", default=10001, help="Ray head port")
@click.option("--db-path", default="~/.wano/wano.db", help="Database path")
def up(port: int, ray_port: int, db_path: str):
    existing_pid = get_pid()
    if existing_pid and is_process_running(existing_pid):
        click.echo(f"Control plane is already running (PID: {existing_pid})", err=True)
        click.echo("Use 'wano down' to stop it first", err=True)
        sys.exit(1)
    db_path_expanded = Path(db_path).expanduser()
    db_path_expanded.parent.mkdir(parents=True, exist_ok=True)
    log_file = Path.home() / ".wano" / "wano.log"
    pid = start_detached(
        [
            sys.executable,
            "-m",
            "wano.control.server_main",
            str(port),
            str(ray_port),
            str(db_path_expanded),
        ],
        log_file,
    )
    save_pid(pid)
    for _ in range(10):
        time.sleep(0.2)
        if not is_process_running(pid):
            click.echo(f"Failed to start control plane. Check logs: {log_file}", err=True)
            if log_file.exists():
                click.echo("\nLast 20 lines of log:", err=True)
                with open(log_file) as f:
                    for line in f.readlines()[-20:]:
                        click.echo(f"  {line.rstrip()}", err=True)
            sys.exit(1)
        try:
            if requests.get(f"http://localhost:{port}/status", timeout=0.5).status_code == 200:
                break
        except Exception:
            pass
    else:
        click.echo(
            f"Warning: Control plane started but not responding yet\nCheck logs: {log_file}",
            err=True,
        )
    click.echo(
        f"Control plane started\nPID: {pid}\nLogs: {log_file}\nAPI server: http://0.0.0.0:{port}\nRay head: port {ray_port}\nJoin token: wano-default-token\n\nUse 'wano down' to stop the control plane"
    )


@cli.command()
@click.option("--control-plane-url", help="Control plane URL (auto-discover if not provided)")
def join(control_plane_url: str):
    agent = NodeAgent(control_plane_url=control_plane_url)
    click.echo("Starting node agent...")
    try:
        agent.start()
    except KeyboardInterrupt:
        click.echo("\nStopping node agent...")
        agent.stop()


@cli.command()
@click.argument("script")
@click.option("--compute", required=True, type=click.Choice(["cpu", "gpu"]), help="Compute type")
@click.option("--gpus", type=int, help="Number of GPUs (for GPU jobs)")
@click.option("--control-plane-url", default="http://localhost:8000", help="Control plane URL")
def run(script: str, compute: str, gpus: int, control_plane_url: str):
    script_path = Path(script)
    if not script_path.exists():
        click.echo(f"Error: Script {script} not found", err=True)
        sys.exit(1)
    exec(script_path.read_text(), globals())
    click.echo("Submitting job...\nNote: Function discovery not fully implemented in MVP")
    import base64

    response = requests.post(
        f"{control_plane_url}/submit",
        json={
            "compute": compute,
            "gpus": gpus,
            "function_code": base64.b64encode(script_path.read_text().encode()).decode(),
        },
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    click.echo(f"Job submitted: {job_id}\nStreaming logs...")
    stream_logs(job_id, control_plane_url)


@cli.command()
@click.option("--control-plane-url", default="http://localhost:8000", help="Control plane URL")
def status(control_plane_url: str):
    try:
        data = requests.get(f"{control_plane_url}/status", timeout=5).json()
        click.echo("Cluster Status:\n\nCompute:")
        compute = data.get("compute", {})
        if "gpu" in compute:
            gpus = compute["gpu"]
            click.echo(f"  GPU: {len(gpus)} available")
            for gpu in gpus:
                if isinstance(gpu, list):
                    node_id = gpu[0].get("node_id", "unknown") if gpu else "unknown"
                    click.echo(f"    {node_id}: {len(gpu)} GPU(s)")
                else:
                    node_id = gpu.get("node_id", "unknown")
                    click.echo(
                        f"    {node_id}: {gpu.get('name', 'GPU')} ({gpu.get('memory_gb', '?')} GB)"
                    )
        if "cpu" in compute:
            cpus = compute["cpu"]
            click.echo(f"  CPU: {len(cpus)} available")
            for cpu in cpus:
                node_id = cpu.get("node_id", "unknown")
                click.echo(
                    f"    {node_id}: {cpu.get('cores', '?')} cores, {cpu.get('memory_gb', '?')} GB RAM"
                )
        click.echo("\nJobs:")
        jobs = data.get("jobs", [])
        if jobs:
            for job in jobs:
                click.echo(
                    f"  {job['job_id'][:8]}... {job['compute']} {job.get('gpus', '')} {job['status']}"
                )
        else:
            click.echo("  No jobs")
    except requests.exceptions.ConnectionError:
        click.echo(
            f"Error: Could not connect to control plane\n  Is the control plane running at {control_plane_url}?\n  Start it with: wano up",
            err=True,
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("job_id")
@click.option("--control-plane-url", default="http://localhost:8000", help="Control plane URL")
def logs(job_id: str, control_plane_url: str):
    try:
        stream_logs(job_id, control_plane_url)
    except requests.exceptions.ConnectionError:
        click.echo(
            f"Error: Could not connect to control plane\n  Is the control plane running at {control_plane_url}?\n  Start it with: wano up",
            err=True,
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def down():
    pid = get_pid()
    if not pid:
        click.echo("Control plane is not running (no PID file found)", err=True)
        sys.exit(1)
    if not is_process_running(pid):
        click.echo(f"Process {pid} is not running (stale PID file)", err=True)
        pid_file = get_pid_file()
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(1)
    click.echo(f"Stopping control plane (PID: {pid})...")
    if kill_process(pid):
        time.sleep(1)
        if is_process_running(pid):
            click.echo("Force killing process...")
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        pid_file = get_pid_file()
        if pid_file.exists():
            pid_file.unlink()
        click.echo("Control plane stopped")
    else:
        click.echo("Failed to stop control plane", err=True)
        sys.exit(1)
