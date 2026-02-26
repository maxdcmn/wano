from dataclasses import dataclass


@dataclass
class ResourceQuota:
    namespace: str
    max_cpu_jobs: int | None = None
    max_gpu_jobs: int | None = None
