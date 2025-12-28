import contextlib
import json
import platform
import socket
import subprocess
import warnings

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
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import psutil

from wano.models.compute import CPUSpec, GPUSpec, NodeCapabilities


def detect_gpus() -> list[GPUSpec]:
    gpus = []
    if HAS_NVML:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
                pynvml.nvmlInit()
                for i in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    fan = None
                    power_usage = None
                    power_cap = None
                    utilization = None
                    memory_used = None
                    with contextlib.suppress(NVMLError):
                        fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                    with contextlib.suppress(NVMLError):
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
                        power_cap = (
                            pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] // 1000
                        )
                    with contextlib.suppress(NVMLError):
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization = float(util.gpu)
                    mem_info = None
                    memory_used = None
                    with contextlib.suppress(NVMLError):
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_used = int(mem_info.used // (1024**2))
                    gpus.append(
                        GPUSpec(
                            name=pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
                            memory_gb=mem_info.total // (1024**3) if mem_info else 0,
                            fan_percent=fan,
                            power_usage_w=power_usage,
                            power_cap_w=power_cap,
                            utilization_percent=utilization,
                            memory_used_mib=memory_used,
                        )
                    )
                pynvml.nvmlShutdown()
            return gpus
        except (NVMLError, RuntimeError):
            pass
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUSpec(name=props.name, memory_gb=props.total_memory // (1024**3)))
        return gpus
    return []


def _get_cpu_metrics() -> tuple[float | None, float | None, float | None]:
    temp = power = power_max = None
    try:
        if platform.system() == "Darwin":
            with contextlib.suppress(ImportError, Exception):
                from macmon import MacMon

                m = MacMon().get_metrics()
                metrics = json.loads(m) if isinstance(m, str) else m
                if metrics:
                    if (
                        "temp" in metrics
                        and (t := metrics["temp"].get("cpu_temp_avg"))
                        and 0 < t < 150
                    ):
                        temp = float(t)
                    if "cpu_power" in metrics and (p := metrics["cpu_power"]) and p > 0:
                        power = float(p)
                    if (
                        isinstance(soc := metrics.get("soc"), dict)
                        and "pcpu_cores" in soc
                        and "ecpu_cores" in soc
                        and (total := soc.get("pcpu_cores", 0) + soc.get("ecpu_cores", 0)) > 0
                    ):
                        power_max = total * 5.0
    except (ImportError, RuntimeError, json.JSONDecodeError):
        pass
    if temp is None and psutil:
        with contextlib.suppress(Exception):
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for entries in temps.values():
                        if entries and (t := entries[0].current) and t > 0:
                            temp = float(t)
                            break
    return temp, power, power_max


def detect_cpu() -> CPUSpec:
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    cpu_name = None
    if platform.system() == "Darwin":
        with contextlib.suppress(Exception):
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0 and result.stdout.strip():
                cpu_name = result.stdout.strip()
    elif platform.system() == "Linux":
        with contextlib.suppress(Exception), open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
                    break
    if not cpu_name:
        cpu_name = platform.processor() or platform.machine()
    temp, power, power_max = _get_cpu_metrics()
    utilization = None
    memory_used = None
    if psutil:
        with contextlib.suppress(Exception):
            utilization = psutil.cpu_percent(interval=0.1)
        with contextlib.suppress(Exception):
            mem = psutil.virtual_memory()
            memory_used = int(mem.used // (1024**2))
    return CPUSpec(
        cores=cores,
        memory_gb=psutil.virtual_memory().total // (1024**3),
        name=cpu_name,
        temp_celsius=temp,
        power_usage_w=power,
        power_cap_w=power_max,
        utilization_percent=utilization,
        memory_used_mib=memory_used,
    )


def detect_all() -> NodeCapabilities:
    compute: dict[str, list[GPUSpec] | CPUSpec] = {}
    gpus = detect_gpus()
    if gpus:
        compute["gpu"] = gpus
    compute["cpu"] = detect_cpu()
    return NodeCapabilities(node_id=socket.gethostname(), compute=compute)
