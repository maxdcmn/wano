from conftest import make_job

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
    db.create_job(make_job("job1"))
    db.assign_job("job1", ["node1"])
    db.complete_job("job1")
    job = db.get_job("job1")
    assert job.status == JobStatus.COMPLETED
    assert job.error is None


def test_job_fails_with_error(db):
    db.create_job(make_job("job2"))
    db.assign_job("job2", ["node1"])
    db.complete_job("job2", "error")
    job = db.get_job("job2")
    assert job.status == JobStatus.FAILED
    assert job.error == "error"


def test_job_stores_result(db):
    db.create_job(make_job("job3"))
    db.assign_job("job3", ["node1"])
    db.complete_job("job3", result='{"value": 42}')
    job = db.get_job("job3")
    assert job.status == JobStatus.COMPLETED
    assert job.result == '{"value": 42}'


def test_get_job_returns_result(db):
    db.create_job(make_job("job4"))
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


def test_get_pending_jobs(db):
    db.create_job(make_job("job1"))
    db.create_job(make_job("job2"))
    db.assign_job("job2", ["node1"])
    pending = db.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == "job1"
    assert pending[0].status == JobStatus.PENDING


def test_job_priority_ordering(db):
    db.create_job(make_job("job1", priority=0))
    db.create_job(make_job("job2", priority=5))
    pending = db.get_pending_jobs()
    assert pending[0].job_id == "job2"


def test_attempts_increment(db):
    db.create_job(make_job("job1"))
    db.assign_job("job1", ["node1"])
    job = db.get_job("job1")
    assert job.attempts == 1


def test_job_retries_when_allowed(db):
    db.create_job(make_job("job1", max_retries=1))
    db.assign_job("job1", ["node1"])
    db.complete_job("job1", "error")
    job = db.get_job("job1")
    assert job.status == JobStatus.PENDING
    assert job.attempts == 1


def test_job_fails_after_retries_exhausted(db):
    db.create_job(make_job("job1", max_retries=0))
    db.assign_job("job1", ["node1"])
    db.complete_job("job1", "error")
    job = db.get_job("job1")
    assert job.status == JobStatus.FAILED


def test_cancel_job(db):
    db.create_job(make_job("job1"))
    db.assign_job("job1", ["node1"])
    db.cancel_job("job1")
    job = db.get_job("job1")
    assert job.status == JobStatus.CANCELLED


def test_job_env_vars_persisted(db):
    db.create_job(make_job("job-env", env_vars='{"FOO": "bar"}'))
    job = db.get_job("job-env")
    assert job.env_vars == '{"FOO": "bar"}'


def test_get_node_usage(db):
    db.create_job(make_job("job1"))
    db.assign_job("job1", ["node1"])
    usage = db.get_node_usage()
    assert usage["node1"]["cpu"] == 1


def test_job_timeout_persisted(db):
    db.create_job(make_job("job-timeout", timeout_seconds=300))
    job = db.get_job("job-timeout")
    assert job.timeout_seconds == 300


def test_job_timeout_none_by_default(db):
    db.create_job(make_job("job-no-timeout"))
    job = db.get_job("job-no-timeout")
    assert job.timeout_seconds is None


def test_fail_job_timeout_skips_retries(db):
    db.create_job(make_job("job-t", max_retries=5, timeout_seconds=10))
    db.assign_job("job-t", ["node1"])
    db.fail_job_timeout("job-t", error="Job timed out after 10 seconds")
    job = db.get_job("job-t")
    assert job.status == JobStatus.FAILED
    assert "timed out" in job.error


def test_get_running_jobs(db):
    db.create_job(make_job("job-a"))
    db.create_job(make_job("job-b"))
    db.assign_job("job-a", ["node1"])
    running = db.get_running_jobs()
    assert len(running) == 1
    assert running[0].job_id == "job-a"


def test_get_running_jobs_with_timeout(db):
    db.create_job(make_job("job-rt", timeout_seconds=60))
    db.assign_job("job-rt", ["node1"])
    running = db.get_running_jobs()
    assert running[0].timeout_seconds == 60


def test_job_depends_on_persisted(db):
    db.create_job(make_job("dep1"))
    db.create_job(make_job("job-with-deps", depends_on=["dep1"]))
    job = db.get_job("job-with-deps")
    assert job.depends_on == ["dep1"]


def test_job_depends_on_none_by_default(db):
    db.create_job(make_job("job-no-deps"))
    job = db.get_job("job-no-deps")
    assert job.depends_on is None


def test_get_pending_jobs_skips_unsatisfied_deps(db):
    db.create_job(make_job("dep1"))
    db.create_job(make_job("job-blocked", depends_on=["dep1"]))
    pending = db.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == "dep1"


def test_get_pending_jobs_includes_satisfied_deps(db):
    db.create_job(make_job("dep1"))
    db.assign_job("dep1", ["node1"])
    db.complete_job("dep1")
    db.create_job(make_job("job-ready", depends_on=["dep1"]))
    pending = db.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == "job-ready"


def test_cascade_failure_direct(db):
    db.create_job(make_job("dep1"))
    db.create_job(make_job("job-child", depends_on=["dep1"]))
    db.assign_job("dep1", ["node1"])
    db.complete_job("dep1", error="boom")
    db.cascade_failure("dep1")
    child = db.get_job("job-child")
    assert child.status == JobStatus.FAILED
    assert "dep1" in child.error


def test_cascade_failure_recursive(db):
    db.create_job(make_job("a"))
    db.create_job(make_job("b", depends_on=["a"]))
    db.create_job(make_job("c", depends_on=["b"]))
    db.assign_job("a", ["node1"])
    db.complete_job("a", error="boom")
    db.cascade_failure("a")
    assert db.get_job("b").status == JobStatus.FAILED
    assert db.get_job("c").status == JobStatus.FAILED
    assert "a" in db.get_job("b").error
    assert "b" in db.get_job("c").error


def test_cascade_failure_from_cancel(db):
    db.create_job(make_job("dep1"))
    db.create_job(make_job("child1", depends_on=["dep1"]))
    db.assign_job("dep1", ["node1"])
    db.cancel_job("dep1")
    db.cascade_failure("dep1")
    child = db.get_job("child1")
    assert child.status == JobStatus.FAILED
    assert "dep1" in child.error


def test_validate_depends_on_missing(db):
    db.create_job(make_job("exists"))
    missing = db.validate_depends_on(["exists", "does-not-exist"])
    assert missing == ["does-not-exist"]


def test_validate_depends_on_all_valid(db):
    db.create_job(make_job("a"))
    db.create_job(make_job("b"))
    assert db.validate_depends_on(["a", "b"]) == []


def test_cascade_does_not_affect_completed(db):
    db.create_job(make_job("dep1"))
    db.create_job(make_job("already-done", depends_on=["dep1"]))
    db.assign_job("already-done", ["node1"])
    db.complete_job("already-done")
    db.assign_job("dep1", ["node1"])
    db.complete_job("dep1", error="boom")
    db.cascade_failure("dep1")
    assert db.get_job("already-done").status == JobStatus.COMPLETED


def test_cordoned_node_remains_cordoned(db):
    capabilities = NodeCapabilities(
        node_id="node1", compute={"cpu": CPUSpec(cores=8, memory_gb=16)}
    )
    db.register_node("node1", capabilities)
    assert db.set_node_status("node1", "cordoned")
    db.register_node("node1", capabilities)
    nodes = {n["node_id"]: n["status"] for n in db.get_nodes()}
    assert nodes["node1"] == "cordoned"


def test_node_labels_persisted(db):
    capabilities = NodeCapabilities(
        node_id="node1",
        compute={"cpu": CPUSpec(cores=4, memory_gb=8)},
        labels={"rack": "A", "env": "prod"},
    )
    db.register_node("node1", capabilities)
    labels = db.get_node_labels()
    assert labels["node1"] == {"rack": "A", "env": "prod"}


def test_job_node_selector_persisted(db):
    db.create_job(make_job("job-sel", node_selector={"rack": "A"}))
    job = db.get_job("job-sel")
    assert job.node_selector == {"rack": "A"}


def test_job_node_selector_none_by_default(db):
    db.create_job(make_job("job-no-sel"))
    job = db.get_job("job-no-sel")
    assert job.node_selector is None


def test_job_namespace_persisted(db):
    db.create_job(make_job("job-ns", namespace="team-a"))
    job = db.get_job("job-ns")
    assert job.namespace == "team-a"


def test_job_namespace_none_by_default(db):
    db.create_job(make_job("job-no-ns"))
    job = db.get_job("job-no-ns")
    assert job.namespace is None


def test_create_and_get_quota(db):
    db.create_or_update_quota("team-a", max_cpu_jobs=5, max_gpu_jobs=2)
    quota = db.get_quota("team-a")
    assert quota is not None
    assert quota.namespace == "team-a"
    assert quota.max_cpu_jobs == 5
    assert quota.max_gpu_jobs == 2


def test_get_namespace_usage(db):
    db.create_job(make_job("job1", namespace="team-a"))
    db.create_job(make_job("job2", "gpu", gpus=1, namespace="team-a"))
    db.assign_job("job1", ["node1"])
    db.assign_job("job2", ["node1"])
    usage = db.get_namespace_usage("team-a")
    assert usage["cpu"] == 1
    assert usage["gpu"] == 1


def test_delete_quota(db):
    db.create_or_update_quota("team-b", max_cpu_jobs=3)
    assert db.delete_quota("team-b")
    assert db.get_quota("team-b") is None
    assert not db.delete_quota("nonexistent")
