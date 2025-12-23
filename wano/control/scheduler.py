from wano.models.job import Job


class Scheduler:
    def schedule_job(self, job: Job, available_compute: dict[str, list[dict]]) -> list[str] | None:
        if job.compute == "cpu":
            return (
                [self._get_nodes_with_compute(available_compute, "cpu")[0]]
                if "cpu" in available_compute
                and self._get_nodes_with_compute(available_compute, "cpu")
                else None
            )
        elif job.compute == "gpu":
            return self._schedule_gpu_job(job, available_compute)
        return None

    def _schedule_gpu_job(
        self, job: Job, available_compute: dict[str, list[dict]]
    ) -> list[str] | None:
        if "gpu" not in available_compute:
            return None
        gpus_needed = job.gpus or 1
        node_gpus = self._get_node_gpu_map(available_compute)
        selected_nodes = []
        gpus_assigned = 0
        for _gpu_model, nodes_with_model in self._group_by_gpu_model(node_gpus).items():
            if gpus_assigned >= gpus_needed:
                break
            for node_id, gpu_count in nodes_with_model:
                if gpus_assigned >= gpus_needed:
                    break
                if node_id not in selected_nodes:
                    selected_nodes.append(node_id)
                gpus_assigned += min(gpu_count, gpus_needed - gpus_assigned)
        if gpus_assigned < gpus_needed:
            for node_id, gpu_list in node_gpus.items():
                if gpus_assigned >= gpus_needed:
                    break
                if node_id not in selected_nodes:
                    selected_nodes.append(node_id)
                    gpus_assigned += len(gpu_list)
        return selected_nodes if gpus_assigned >= gpus_needed else None

    def _get_nodes_with_compute(
        self, available_compute: dict[str, list[dict]], compute_type: str
    ) -> list[str]:
        return []

    def _get_node_gpu_map(self, available_compute: dict[str, list[dict]]) -> dict[str, list[dict]]:
        return {}

    def _group_by_gpu_model(self, node_gpus: dict[str, list[dict]]) -> dict[str, list[tuple]]:
        model_to_nodes: dict[str, list[tuple]] = {}
        for node_id, gpu_list in node_gpus.items():
            for gpu in gpu_list:
                model = gpu.get("name", "unknown")
                model_to_nodes.setdefault(model, []).append((node_id, 1))
        return model_to_nodes
