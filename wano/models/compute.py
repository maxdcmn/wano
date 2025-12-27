from dataclasses import dataclass


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

    def to_dict(self) -> dict:
        result: dict = {"node_id": self.node_id, "compute": {}}
        if "gpu" in self.compute and isinstance(self.compute["gpu"], list):
            result["compute"]["gpu"] = [
                {
                    "name": g.name,
                    "memory_gb": g.memory_gb,
                    "fan_percent": g.fan_percent,
                    "power_usage_w": g.power_usage_w,
                    "power_cap_w": g.power_cap_w,
                    "utilization_percent": g.utilization_percent,
                    "memory_used_mib": g.memory_used_mib,
                }
                for g in self.compute["gpu"]
            ]
        if "cpu" in self.compute and isinstance(self.compute["cpu"], CPUSpec):
            c = self.compute["cpu"]
            result["compute"]["cpu"] = {
                "cores": c.cores,
                "memory_gb": c.memory_gb,
                "name": c.name,
                "temp_celsius": c.temp_celsius,
                "power_usage_w": c.power_usage_w,
                "power_cap_w": c.power_cap_w,
                "utilization_percent": c.utilization_percent,
                "memory_used_mib": c.memory_used_mib,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "NodeCapabilities":
        compute: dict[str, list[GPUSpec] | CPUSpec] = {}
        if "gpu" in data.get("compute", {}):
            compute["gpu"] = [
                GPUSpec(
                    name=g["name"],
                    memory_gb=g["memory_gb"],
                    fan_percent=g.get("fan_percent"),
                    power_usage_w=g.get("power_usage_w"),
                    power_cap_w=g.get("power_cap_w"),
                    utilization_percent=g.get("utilization_percent"),
                    memory_used_mib=g.get("memory_used_mib"),
                )
                for g in data["compute"]["gpu"]
            ]
        if "cpu" in data.get("compute", {}):
            c = data["compute"]["cpu"]
            compute["cpu"] = CPUSpec(
                cores=c["cores"],
                memory_gb=c["memory_gb"],
                name=c.get("name"),
                temp_celsius=c.get("temp_celsius"),
                power_usage_w=c.get("power_usage_w"),
                power_cap_w=c.get("power_cap_w"),
                utilization_percent=c.get("utilization_percent"),
                memory_used_mib=c.get("memory_used_mib"),
            )
        return cls(node_id=data["node_id"], compute=compute)
