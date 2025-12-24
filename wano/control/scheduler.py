from wano.models.job import Job


class Scheduler:
    def schedule_job(self, job: Job, available_compute: dict[str, list[dict]]) -> list[str] | None:
        if job.compute == "cpu":
            if "cpu" in available_compute and available_compute["cpu"]:
                return [available_compute["cpu"][0].get("node_id", "unknown")]
            return None
        elif job.compute == "gpu":
            return self._schedule_gpu_job(job, available_compute)
        return None

    def _schedule_gpu_job(
        self, job: Job, available_compute: dict[str, list[dict]]
    ) -> list[str] | None:
        if "gpu" not in available_compute:
            return None
        gpus_needed = job.gpus or 1
        node_gpus: dict[str, list[dict]] = {}
        for gpu_entry in available_compute["gpu"]:
            if isinstance(gpu_entry, list):
                node_id = gpu_entry[0].get("node_id") if gpu_entry else None
                if node_id:
                    node_gpus.setdefault(node_id, []).extend(gpu_entry)
            else:
                node_id = gpu_entry.get("node_id")
                if node_id:
                    node_gpus.setdefault(node_id, []).append(gpu_entry)
        selected_nodes = []
        gpus_assigned = 0
        for node_id, gpu_list in node_gpus.items():
            if gpus_assigned >= gpus_needed:
                break
            if node_id not in selected_nodes:
                selected_nodes.append(node_id)
            gpus_assigned += len(gpu_list)
        return selected_nodes if gpus_assigned >= gpus_needed else None
