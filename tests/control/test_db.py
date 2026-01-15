from wano.models.compute import CPUSpec, GPUSpec, NodeCapabilities
from wano.models.job import JobStatus


def test_register_node_and_get_compute(db):
    capabilities = NodeCapabilities(
        node_id="node1",
        compute={
            "cpu": CPUSpec(cores=8, memory_gb=16),
            "gpu": [GPUSpec(name="GPU1", memory_gb=24)],
        },
    )
    db.register_node("node1", capabilities)

    assert "node1" in db.get_active_nodes()
    assert "cpu" in db.get_available_compute()
    assert "gpu" in db.get_available_compute()


def test_job_completes_successfully(db):
    job = db.create_job("job1", "cpu", None, "def f(): pass")
    db.assign_job("job1", ["node1"])
    db.complete_job("job1")
    job = db.get_job("job1")
    assert job.status == JobStatus.COMPLETED
    assert job.error is None


def test_job_fails_with_error(db):
    job = db.create_job("job2", "cpu", None, "def f(): pass")
    db.assign_job("job2", ["node1"])
    db.complete_job("job2", "error")
    job = db.get_job("job2")
    assert job.status == JobStatus.FAILED
    assert job.error == "error"


def test_job_stores_result(db):
    job = db.create_job("job3", "cpu", None, "def f(): pass")
    db.assign_job("job3", ["node1"])
    db.complete_job("job3", result='{"value": 42}')
    job = db.get_job("job3")
    assert job.status == JobStatus.COMPLETED
    assert job.result == '{"value": 42}'


def test_get_job_returns_result(db):
    job = db.create_job("job4", "cpu", None, "def f(): pass")
    db.assign_job("job4", ["node1"])
    db.complete_job("job4", result='{"result": "success"}')
    job = db.get_job("job4")
    assert job.result == '{"result": "success"}'


def test_heartbeat_keeps_node_active(db):
    capabilities = NodeCapabilities(
        node_id="node1", compute={"cpu": CPUSpec(cores=8, memory_gb=16)}
    )
    db.register_node("node1", capabilities)
    assert "node1" in db.get_active_nodes(heartbeat_timeout_seconds=1)
    db.update_heartbeat("node1")
    assert "node1" in db.get_active_nodes(heartbeat_timeout_seconds=1)


def test_heartbeat_timeout(db):
    capabilities = NodeCapabilities(
        node_id="node1", compute={"cpu": CPUSpec(cores=8, memory_gb=16)}
    )
    db.register_node("node1", capabilities)
    assert "node1" not in db.get_active_nodes(heartbeat_timeout_seconds=0)


def test_cancel_job(db):
    job = db.create_job("job1", "cpu", None, "def f(): pass")
    db.assign_job("job1", ["node1"])
    db.cancel_job("job1")
    job = db.get_job("job1")
    assert job.status == JobStatus.CANCELLED
