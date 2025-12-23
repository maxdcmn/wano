from dataclasses import dataclass


@dataclass
class GPUSpec:
    name: str
    memory_gb: int


@dataclass
class CPUSpec:
    cores: int
    memory_gb: int


@dataclass
class NodeCapabilities:
    node_id: str
    compute: dict[str, list[GPUSpec] | CPUSpec]

    def to_dict(self) -> dict:
        result: dict = {"node_id": self.node_id, "compute": {}}
        if "gpu" in self.compute and isinstance(self.compute["gpu"], list):
            result["compute"]["gpu"] = [
                {"name": g.name, "memory_gb": g.memory_gb} for g in self.compute["gpu"]
            ]
        if "cpu" in self.compute and isinstance(self.compute["cpu"], CPUSpec):
            c = self.compute["cpu"]
            result["compute"]["cpu"] = {"cores": c.cores, "memory_gb": c.memory_gb}
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "NodeCapabilities":
        compute: dict[str, list[GPUSpec] | CPUSpec] = {}
        if "gpu" in data.get("compute", {}):
            compute["gpu"] = [
                GPUSpec(name=g["name"], memory_gb=g["memory_gb"]) for g in data["compute"]["gpu"]
            ]
        if "cpu" in data.get("compute", {}):
            c = data["compute"]["cpu"]
            compute["cpu"] = CPUSpec(cores=c["cores"], memory_gb=c["memory_gb"])
        return cls(node_id=data["node_id"], compute=compute)
