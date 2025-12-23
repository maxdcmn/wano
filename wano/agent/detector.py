import socket
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
    try:
        import pynvml

        HAS_NVML = True
    except ImportError:
        HAS_NVML = False

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
                    gpus.append(
                        GPUSpec(
                            name=pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
                            memory_gb=pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024**3),
                        )
                    )
                pynvml.nvmlShutdown()
            return gpus
        except Exception:
            pass
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUSpec(name=props.name, memory_gb=props.total_memory // (1024**3)))
        return gpus
    return []


def detect_cpu() -> CPUSpec:
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    return CPUSpec(cores=cores, memory_gb=psutil.virtual_memory().total // (1024**3))


def detect_all() -> NodeCapabilities:
    compute: dict[str, list[GPUSpec] | CPUSpec] = {}
    gpus = detect_gpus()
    if gpus:
        compute["gpu"] = gpus
    compute["cpu"] = detect_cpu()
    return NodeCapabilities(node_id=socket.gethostname(), compute=compute)
