from typing import Any

from wano.models.job import Job


class Scheduler:
    def schedule_job(self, job: Job, available_compute: dict[str, list[Any]]) -> list[str] | None:
        if job.compute == "cpu":
            if "cpu" in available_compute and available_compute["cpu"]:
                return [available_compute["cpu"][0].get("node_id", "unknown")]
            return None
        elif job.compute == "gpu":
            return self._schedule_gpu_job(job, available_compute)
        return None

    def _schedule_gpu_job(
        self, job: Job, available_compute: dict[str, list[Any]]
    ) -> list[str] | None:
        if "gpu" not in available_compute:
            return None
        gpus_needed = job.gpus or 1
        node_gpus: dict[str, int] = {}
        for gpu_entry in available_compute["gpu"]:
            if isinstance(gpu_entry, list):
                node_id = gpu_entry[0].get("node_id") if gpu_entry else None
                if node_id:
                    node_gpus[node_id] = node_gpus.get(node_id, 0) + len(gpu_entry)
            else:
                node_id = gpu_entry.get("node_id")
                if node_id:
                    node_gpus[node_id] = node_gpus.get(node_id, 0) + 1
        assignments: list[str] = []
        for node_id, count in node_gpus.items():
            remaining = gpus_needed - len(assignments)
            if remaining <= 0:
                break
            assign = min(count, remaining)
            assignments.extend([node_id] * assign)
        return assignments if len(assignments) >= gpus_needed else None
