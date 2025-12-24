# Wano

Local-first compute orchestration for multi-node CPU/GPU workloads. Request abstract compute types, not specific devices. Automatic node discovery, distributed execution via Ray, self-hosted with no cloud dependency.

## CLI Commands

- `wano up` - Start the control plane (runs detached by default)
- `wano down` - Stop the control plane
- `wano join` - Register this machine as a worker node
- `wano status` - View cluster status and active jobs
- `wano run <script> --compute <cpu|gpu> [--gpus N]` - Submit and execute a job

## Python API

```python
import wano

@wano.function(compute="gpu", gpus=4)
def train():
    # Runs on 4 GPUs, potentially across multiple nodes
    ...

@wano.function(compute="cpu")
def preprocess():
    # Runs on CPU resources
    ...
```
