import os
import signal
import subprocess
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
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=False,
            )
            return process.pid
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    return process.pid
