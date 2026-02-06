from typing import Any

from wano.models.job import Job


class Scheduler:
    def schedule_job(
        self,
        job: Job,
        available_compute: dict[str, list[Any]],
        node_usage: dict[str, dict[str, int]] | None = None,
    ) -> list[str] | None:
        if job.compute == "cpu":
            return self._schedule_cpu_job(available_compute, node_usage or {})
        elif job.compute == "gpu":
            return self._schedule_gpu_job(job, available_compute, node_usage or {})
        return None

    def _schedule_cpu_job(
        self, available_compute: dict[str, list[Any]], node_usage: dict[str, dict[str, int]]
    ) -> list[str] | None:
        if "cpu" not in available_compute:
            return None
        for cpu in available_compute["cpu"]:
            node_id = cpu.get("node_id") if isinstance(cpu, dict) else None
            cores = cpu.get("cores", 0) if isinstance(cpu, dict) else 0
            if not node_id or cores <= 0:
                continue
            used = node_usage.get(node_id, {}).get("cpu", 0)
            if cores - used >= 1:
                return [node_id]
        return None

    def _schedule_gpu_job(
        self,
        job: Job,
        available_compute: dict[str, list[Any]],
        node_usage: dict[str, dict[str, int]],
    ) -> list[str] | None:
        if "gpu" not in available_compute:
            return None
        gpus_needed = job.gpus or 1
        node_gpus: dict[str, int] = {}
        node_order: list[str] = []
        for gpu_entry in available_compute["gpu"]:
            if isinstance(gpu_entry, list):
                node_id = gpu_entry[0].get("node_id") if gpu_entry else None
                if node_id:
                    if node_id not in node_gpus:
                        node_order.append(node_id)
                    node_gpus[node_id] = node_gpus.get(node_id, 0) + len(gpu_entry)
            else:
                node_id = gpu_entry.get("node_id")
                if node_id:
                    if node_id not in node_gpus:
                        node_order.append(node_id)
                    node_gpus[node_id] = node_gpus.get(node_id, 0) + 1
        assignments: list[str] = []
        for node_id in node_order:
            count = node_gpus.get(node_id, 0)
            used = node_usage.get(node_id, {}).get("gpu", 0)
            free = max(0, count - used)
            remaining = gpus_needed - len(assignments)
            if remaining <= 0:
                break
            assign = min(free, remaining)
            assignments.extend([node_id] * assign)
        return assignments if len(assignments) >= gpus_needed else None
