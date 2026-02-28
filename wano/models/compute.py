from dataclasses import asdict, dataclass


@dataclass
class GPUSpec:
    name: str
    memory_gb: int
    fan_percent: int | None = None
    power_usage_w: int | None = None
    power_cap_w: int | None = None
    utilization_percent: float | None = None
    memory_used_mib: int | None = None


@dataclass
class CPUSpec:
    cores: int
    memory_gb: int
    name: str | None = None
    temp_celsius: float | None = None
    power_usage_w: float | None = None
    power_cap_w: float | None = None
    utilization_percent: float | None = None
    memory_used_mib: int | None = None


@dataclass
class NodeCapabilities:
    node_id: str
    compute: dict[str, list[GPUSpec] | CPUSpec]
    ray_node_id: str | None = None
    labels: dict[str, str] | None = None

    def to_dict(self) -> dict:
        result: dict = {"node_id": self.node_id, "compute": {}}
        if self.ray_node_id:
            result["ray_node_id"] = self.ray_node_id
        if self.labels:
            result["labels"] = self.labels
        for key, spec in self.compute.items():
            if isinstance(spec, list):
                result["compute"][key] = [asdict(g) for g in spec]
            else:
                result["compute"][key] = asdict(spec)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "NodeCapabilities":
        compute: dict[str, list[GPUSpec] | CPUSpec] = {}
        raw = data.get("compute", {})
        if "gpu" in raw:
            compute["gpu"] = [GPUSpec(**g) for g in raw["gpu"]]
        if "cpu" in raw:
            compute["cpu"] = CPUSpec(**raw["cpu"])
        return cls(
            node_id=data["node_id"],
            compute=compute,
            ray_node_id=data.get("ray_node_id"),
            labels=data.get("labels"),
        )
