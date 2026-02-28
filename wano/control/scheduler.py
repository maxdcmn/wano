from __future__ import annotations

from typing import Any

from wano.models.job import Job
from wano.models.quota import ResourceQuota


class Scheduler:
    def schedule_job(
        self,
        job: Job,
        available_compute: dict[str, list[Any]],
        node_usage: dict[str, dict[str, int]] | None = None,
        node_labels: dict[str, dict[str, str]] | None = None,
        quota: ResourceQuota | None = None,
        namespace_usage: dict[str, int] | None = None,
    ) -> list[str] | None:
        if (
            quota
            and namespace_usage is not None
            and not self._check_quota(job, quota, namespace_usage)
        ):
            return None
        compute = available_compute
        if job.node_selector:
            compute = self._filter_by_labels(compute, job.node_selector, node_labels or {})
        if job.compute == "cpu":
            return self._schedule_cpu_job(compute, node_usage or {})
        elif job.compute == "gpu":
            return self._schedule_gpu_job(job, compute, node_usage or {})
        return None

    def _check_quota(self, job: Job, quota: ResourceQuota, namespace_usage: dict[str, int]) -> bool:
        if job.compute == "cpu" and quota.max_cpu_jobs is not None:
            return namespace_usage.get("cpu", 0) < quota.max_cpu_jobs
        if job.compute == "gpu" and quota.max_gpu_jobs is not None:
            return namespace_usage.get("gpu", 0) < quota.max_gpu_jobs
        return True

    def _filter_by_labels(
        self,
        available_compute: dict[str, list[Any]],
        node_selector: dict[str, str],
        node_labels: dict[str, dict[str, str]],
    ) -> dict[str, list[Any]]:
        def _matches(node_id: str) -> bool:
            labels = node_labels.get(node_id, {})
            return all(labels.get(k) == v for k, v in node_selector.items())

        filtered: dict[str, list[Any]] = {}
        for compute_type, entries in available_compute.items():
            kept = []
            for entry in entries:
                if isinstance(entry, list):
                    node_id = entry[0].get("node_id") if entry else None
                elif isinstance(entry, dict):
                    node_id = entry.get("node_id")
                else:
                    node_id = None
                if node_id and _matches(node_id):
                    kept.append(entry)
            if kept:
                filtered[compute_type] = kept
        return filtered

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
                count = len(gpu_entry)
            else:
                node_id = gpu_entry.get("node_id")
                count = 1
            if node_id:
                if node_id not in node_gpus:
                    node_order.append(node_id)
                node_gpus[node_id] = node_gpus.get(node_id, 0) + count
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
