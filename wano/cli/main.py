import base64
import contextlib
import json
import os
import signal
import socket
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import click
import requests

import wano
from wano.agent.discovery import discover_control_plane
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

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
    try:
        import pynvml

        HAS_NVML = True
        NVMLError = pynvml.NVMLError
    except ImportError:
        HAS_NVML = False
        NVMLError = type("NVMLError", (Exception,), {})

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


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

    existing = discover_control_plane(timeout=2.0)
    if existing:
        click.echo(f"Control plane already exists on network: {existing}", err=True)
        click.echo("Join it with: wano join", err=True)
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
        except requests.exceptions.RequestException:
            pass
    else:
        click.echo(
            f"Warning: Control plane started but not responding yet\nCheck logs: {log_file}",
            err=True,
        )
    click.echo(
        f"Control plane started\nPID: {pid}\nLogs: {log_file}\nAPI: http://0.0.0.0:{port}\nRay: port {ray_port}\n\nUse 'wano down' to stop"
    )


@cli.command()
@click.option("--control-plane-url", help="Control plane URL (auto-discover if not provided)")
def join(control_plane_url: str):
    agent = NodeAgent(control_plane_url=control_plane_url)
    click.echo("Starting node agent...")
    try:
        agent.start()
    except (RuntimeError, requests.exceptions.ConnectionError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
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
    click.echo("Submitting job...\nNote: Function discovery not fully implemented in MVP")
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


def _progress_bar(used: int, total: int, width: int = 10) -> str:
    if total == 0:
        return "[" + " " * width + "] 0%"
    ratio = used / total
    percent = min(100, int(ratio * 100))
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent}%"


def _get_real_time_cpu_usage() -> float:
    if not psutil:
        return 0.0
    try:
        return psutil.cpu_percent(interval=0.1)
    except (AttributeError, RuntimeError):
        return 0.0


def _get_real_time_gpu_usage() -> list[float]:
    if not HAS_NVML:
        return []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            utilizations = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                with contextlib.suppress(NVMLError):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilizations.append(util.gpu)
            pynvml.nvmlShutdown()
            return utilizations
    except (NVMLError, AttributeError):
        return []


def _get_real_time_gpu_power() -> list[tuple[int | None, int | None]]:
    if not HAS_NVML:
        return []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            power_data = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power_usage = None
                power_cap = None
                with contextlib.suppress(NVMLError):
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
                with contextlib.suppress(NVMLError):
                    power_cap = (
                        pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] // 1000
                    )
                power_data.append((power_usage, power_cap))
            pynvml.nvmlShutdown()
            return power_data
    except (NVMLError, AttributeError):
        return []


def _is_current_node(node_id: str, current_hostname: str) -> bool:
    return (
        node_id == current_hostname
        or node_id.startswith(current_hostname)
        or current_hostname.startswith(node_id)
    )


def _parse_node_ids(node_ids_raw: str | list[str] | None) -> list[str]:
    if not node_ids_raw:
        return []
    if isinstance(node_ids_raw, str):
        parsed = json.loads(node_ids_raw)
        return parsed if isinstance(parsed, list) else []
    return node_ids_raw


def _calc_used(
    node_id: str,
    current_hostname: str,
    total: int,
    real_time_data: float | list[float],
    compute_type: str,
    node_usage: dict[str, dict[str, int]],
) -> int:
    if _is_current_node(node_id, current_hostname):
        if (
            compute_type == "gpu"
            and isinstance(real_time_data, list)
            and len(real_time_data) >= total
        ):
            return int(sum(real_time_data[:total]) / 100.0) if total > 0 else 0
        elif compute_type == "cpu" and isinstance(real_time_data, int | float) and psutil:
            return int((real_time_data / 100.0) * total)
    return node_usage.get(node_id, {}).get(compute_type, 0)


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _format_memory(gb: int | None, used_mib: int | None = None) -> str:
    if not gb:
        return "? MiB / ? MiB"
    total_mib = int(gb * 1024)
    used = used_mib if used_mib is not None else 0
    return f"{used} MiB / {total_mib} MiB"


def _format_power(usage: int | float | None, cap: int | float | None = None) -> str:
    if not usage:
        return "N/A"
    fmt = ".1f" if isinstance(usage, float) else ""
    if cap:
        return f"{usage:{fmt}}W / {cap:{fmt}}W"
    return f"{usage:{fmt}}W"


def _format_temp(temp: float | None) -> str:
    return f"{temp:.0f}°C" if temp is not None else "N/A"


def _handle_connection_error(e: requests.exceptions.RequestException, control_plane_url: str):
    msg = (
        f"Error: Could not connect to control plane\n  Is the control plane running at {control_plane_url}?\n  Start it with: wano up"
        if isinstance(e, requests.exceptions.ConnectionError)
        else f"Error: {e}"
    )
    click.echo(msg, err=True)
    sys.exit(1)


def _sep(chars: str, borders: str, widths_list: list[int] | None = None, char: str = "-") -> str:
    w = widths_list or [4, 4, 5, 8, 6]
    return borders[0] + chars.join(char * (width + 2) for width in w) + borders[-1]


@cli.command()
@click.option("--control-plane-url", default="http://localhost:8000", help="Control plane URL")
def status(control_plane_url: str):
    try:
        data = requests.get(f"{control_plane_url}/status", timeout=5).json()
        jobs = data.get("jobs", [])
        current_hostname = socket.gethostname()
        active_nodes = set(data.get("active_nodes", []))
        is_joined = current_hostname in active_nodes
        compute = data.get("compute", {})
        real_time_cpu = _get_real_time_cpu_usage()
        real_time_gpus = _get_real_time_gpu_usage()
        node_usage: dict[str, dict[str, int]] = {}
        for job in jobs:
            if job.get("status") != "running":
                continue
            compute_type = job.get("compute", "cpu")
            count = job.get("gpus", 1) if compute_type == "gpu" else 1
            for node_id in _parse_node_ids(job.get("node_ids")):
                node_usage.setdefault(node_id, {"cpu": 0, "gpu": 0})[compute_type] += count
        compute_rows = []
        all_node_ids_in_compute = set()
        for compute_type in ["gpu", "cpu"]:
            if compute_type not in compute:
                continue
            items = compute[compute_type]
            if compute_type == "gpu":
                node_map: dict[str, list] = {}
                for gpu in items:
                    if isinstance(gpu, list):
                        node_id = gpu[0].get("node_id", "unknown") if gpu else "unknown"
                        node_map.setdefault(node_id, []).extend(gpu)
                    else:
                        node_id = gpu.get("node_id", "unknown")
                        node_map.setdefault(node_id, []).append(gpu)
                for node_id, gpu_list in node_map.items():
                    all_node_ids_in_compute.add(node_id)
                    gpu, total = gpu_list[0] if gpu_list else {}, len(gpu_list)
                    if gpu.get("utilization_percent") is not None:
                        used = (
                            int((gpu.get("utilization_percent") / 100.0) * total)
                            if total > 0
                            else 0
                        )
                    else:
                        used = _calc_used(
                            node_id, current_hostname, total, real_time_gpus, "gpu", node_usage
                        )
                    gpu_power_usage = gpu.get("power_usage_w")
                    gpu_power_cap = gpu.get("power_cap_w")
                    memory_used = gpu.get("memory_used_mib")
                    compute_rows.append(
                        (
                            node_id,
                            gpu.get("name", "GPU"),
                            f"GPU {_progress_bar(used, total)}",
                            _format_memory(gpu.get("memory_gb"), memory_used),
                            "N/A",
                            _format_power(gpu_power_usage, gpu_power_cap),
                            f"{used}/{total}",
                        )
                    )
            else:
                for cpu in items:
                    node_id, cores = cpu.get("node_id", "unknown"), cpu.get("cores", 0)
                    all_node_ids_in_compute.add(node_id)
                    if cpu.get("utilization_percent") is not None:
                        used = int((cpu.get("utilization_percent") / 100.0) * cores)
                    else:
                        used = _calc_used(
                            node_id, current_hostname, cores, real_time_cpu, "cpu", node_usage
                        )
                    temp = cpu.get("temp_celsius")
                    cpu_power = cpu.get("power_usage_w")
                    cpu_power_max = cpu.get("power_cap_w")
                    memory_used = cpu.get("memory_used_mib")
                    compute_rows.append(
                        (
                            node_id,
                            cpu.get("name") or f"{cores} cores",
                            f"CPU {_progress_bar(used, cores)}",
                            _format_memory(cpu.get("memory_gb"), memory_used),
                            _format_temp(temp),
                            _format_power(cpu_power, cpu_power_max),
                            f"{used}/{cores}",
                        )
                    )

        def _col_width(idx: int, default: int = 0) -> int:
            return max(default, max((len(str(r[idx])) for r in compute_rows), default=0))

        node_name_w = max(10, max(_col_width(0), _col_width(1)))
        type_usage_w = max(15, max(_col_width(2), _col_width(3)))
        status_w, temp_w, power_w = _col_width(6, 6), _col_width(4, 6), _col_width(5, 10)
        last_col_w = max(temp_w + status_w + 1, power_w)
        compute_widths = [node_name_w, type_usage_w, last_col_w]
        sep = _sep("+", "||", compute_widths)
        sep_border = _sep("+", "++", compute_widths)
        header_width = len(sep) - 2
        click.echo(datetime.now().strftime("%a %b %d %H:%M:%S %Y"))
        click.echo("+" + "-" * header_width + "+")
        header_line1 = f"Wano {wano.__version__}"
        header_line2 = f"Node Joined: {'Yes' if is_joined else 'No'}"
        remaining_width = header_width - len(header_line1) - 5
        click.echo(f"| {header_line1:<{len(header_line1)}} | {header_line2:<{remaining_width}} |")
        click.echo(sep_border)
        click.echo(
            f"| {'Node':<{node_name_w}} | {'Type Usage':<{type_usage_w}} | {'Temp':<{temp_w}} {'Status':<{status_w}} |"
        )
        power_header = "Power".ljust(last_col_w)
        click.echo(
            f"| {'Name':<{node_name_w}} | {'Memory-Usage':<{type_usage_w}} | {power_header} |"
        )
        click.echo(_sep("+", "||", compute_widths).replace("-", "="))
        for node_id, name, type_usage, memory, temp, power, status in compute_rows:
            temp_status_str = f"{_truncate(str(temp), temp_w):<{temp_w}} {_truncate(str(status), status_w):<{status_w}}"
            click.echo(
                f"| {_truncate(str(node_id), node_name_w):<{node_name_w}} | "
                f"{_truncate(str(type_usage), type_usage_w):<{type_usage_w}} | "
                f"{temp_status_str:<{last_col_w}} |"
            )
            power_formatted = _truncate(str(power), last_col_w).ljust(last_col_w)
            click.echo(
                f"| {_truncate(str(name), node_name_w):<{node_name_w}} | "
                f"{_truncate(str(memory), type_usage_w):<{type_usage_w}} | "
                f"{power_formatted} |"
            )
        click.echo(sep_border)
        compute_table_width = len(sep)
        job_headers = ["Job ID", "Resources", "Nodes", "Status"]
        if jobs:
            max_job_id = max(len(j["job_id"][:8]) for j in jobs)
            max_status = max(len(j.get("status", "")) for j in jobs)
            max_nodes = max(
                len(
                    ", ".join((node_ids := _parse_node_ids(j.get("node_ids")))[:2])
                    + ("..." if len(node_ids) > 2 else "")
                )
                for j in jobs
            )
        else:
            max_job_id, max_status, max_nodes = 8, 0, 0
        job_widths = [
            max(max_job_id, len(job_headers[0])),
            max(8, len(job_headers[1])),
            max(max_nodes, len(job_headers[2])),
            max(max_status, len(job_headers[3])),
        ]
        target_content_width = compute_table_width - 3 * len(job_headers) - 1
        if sum(job_widths) < target_content_width:
            extra = target_content_width - sum(job_widths)
            job_widths[1] += extra // 3
            job_widths[2] += extra - extra // 3
        job_id_w, resources_w, nodes_w, status_w = job_widths
        sep_jobs = _sep("+", "++", job_widths)
        sep_jobs_header = _sep("+", "||", job_widths).replace("-", "=")
        click.echo("\n" + sep_jobs)
        click.echo(
            f"| {'Job ID':<{job_id_w}} | {'Resources':<{resources_w}} | {'Nodes':<{nodes_w}} | {'Status':<{status_w}} |"
        )
        click.echo(sep_jobs_header)
        for job in jobs:
            node_ids = _parse_node_ids(job.get("node_ids"))
            nodes_str = ", ".join(node_ids[:2]) + ("..." if len(node_ids) > 2 else "")
            compute_type, gpus = job.get("compute", "cpu"), job.get("gpus")
            if compute_type == "gpu" and gpus:
                resources = f"{gpus} GPUs"
            elif compute_type == "gpu":
                resources = "1 GPU"
            else:
                resources = "CPU"
            click.echo(
                f"| {_truncate(job['job_id'][:8], job_id_w):<{job_id_w}} | "
                f"{_truncate(resources, resources_w):<{resources_w}} | "
                f"{_truncate(nodes_str, nodes_w):<{nodes_w}} | "
                f"{_truncate(job.get('status', ''), status_w):<{status_w}} |"
            )
        click.echo(sep_jobs)
        failed_jobs = [j for j in jobs if j.get("status") == "failed" and j.get("error")]
        if failed_jobs:
            click.echo("\nFailed Jobs:")
            for job in failed_jobs:
                error_msg = job.get("error", "")
                first_line = error_msg.split("\n")[0] if error_msg else ""
                truncated_error = _truncate(first_line, 80) if first_line else "Unknown error"
                click.echo(f"{job['job_id'][:8]}: {truncated_error}")
    except requests.exceptions.RequestException as e:
        _handle_connection_error(e, control_plane_url)


@cli.command()
@click.argument("job_id")
@click.option("--control-plane-url", default="http://localhost:8000", help="Control plane URL")
def logs(job_id: str, control_plane_url: str):
    try:
        stream_logs(job_id, control_plane_url)
    except requests.exceptions.RequestException as e:
        _handle_connection_error(e, control_plane_url)


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
