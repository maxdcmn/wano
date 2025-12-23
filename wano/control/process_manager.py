import os
import signal
import subprocess
from io import TextIOWrapper
from pathlib import Path


def get_pid_file() -> Path:
    pid_dir = Path.home() / ".wano"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir / "wano.pid"


def save_pid(pid: int):
    get_pid_file().write_text(str(pid))


def get_pid() -> int | None:
    pid_file = get_pid_file()
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_process(pid: int) -> bool:
    try:
        if is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
            return True
        return False
    except OSError:
        return False


def start_detached(command: list, log_file: Path | None = None) -> int:
    stdout: int | TextIOWrapper
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(log_file, "w")  # noqa: SIM115
        stderr = subprocess.STDOUT
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    return process.pid
