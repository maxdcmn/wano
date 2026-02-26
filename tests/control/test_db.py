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
    job = db.create_job("job1", "cpu", None, None, "def f(): pass")
    db.assign_job("job1", ["node1"])
    db.complete_job("job1")
    job = db.get_job("job1")
    assert job.status == JobStatus.COMPLETED
    assert job.error is None


def test_job_fails_with_error(db):
    job = db.create_job("job2", "cpu", None, None, "def f(): pass")
    db.assign_job("job2", ["node1"])
    db.complete_job("job2", "error")
    job = db.get_job("job2")
    assert job.status == JobStatus.FAILED
    assert job.error == "error"


def test_job_stores_result(db):
    job = db.create_job("job3", "cpu", None, None, "def f(): pass")
    db.assign_job("job3", ["node1"])
    db.complete_job("job3", result='{"value": 42}')
    job = db.get_job("job3")
    assert job.status == JobStatus.COMPLETED
    assert job.result == '{"value": 42}'


def test_get_job_returns_result(db):
    job = db.create_job("job4", "cpu", None, None, "def f(): pass")
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
    db.create_job("job1", "cpu", None, None, "def f(): pass")
    db.create_job("job2", "cpu", None, None, "def f(): pass")
    db.assign_job("job2", ["node1"])
    pending = db.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == "job1"
    assert pending[0].status == JobStatus.PENDING


def test_job_priority_ordering(db):
    db.create_job("job1", "cpu", None, None, "def f(): pass", priority=0)
    db.create_job("job2", "cpu", None, None, "def f(): pass", priority=5)
    pending = db.get_pending_jobs()
    assert pending[0].job_id == "job2"


def test_attempts_increment(db):
    db.create_job("job1", "cpu", None, None, "def f(): pass")
    db.assign_job("job1", ["node1"])
    job = db.get_job("job1")
    assert job.attempts == 1


def test_job_retries_when_allowed(db):
    db.create_job("job1", "cpu", None, None, "def f(): pass", max_retries=1)
    db.assign_job("job1", ["node1"])
    db.complete_job("job1", "error")
    job = db.get_job("job1")
    assert job.status == JobStatus.PENDING
    assert job.attempts == 1


def test_job_fails_after_retries_exhausted(db):
    db.create_job("job1", "cpu", None, None, "def f(): pass", max_retries=0)
    db.assign_job("job1", ["node1"])
    db.complete_job("job1", "error")
    job = db.get_job("job1")
    assert job.status == JobStatus.FAILED


def test_cancel_job(db):
    job = db.create_job("job1", "cpu", None, None, "def f(): pass")
    db.assign_job("job1", ["node1"])
    db.cancel_job("job1")
    job = db.get_job("job1")
    assert job.status == JobStatus.CANCELLED


def test_job_env_vars_persisted(db):
    env_vars = '{"FOO": "bar"}'
    db.create_job("job-env", "cpu", None, None, "def f(): pass", env_vars=env_vars)
    job = db.get_job("job-env")
    assert job.env_vars == env_vars


def test_get_node_usage(db):
    db.create_job("job1", "cpu", None, None, "def f(): pass")
    db.assign_job("job1", ["node1"])
    usage = db.get_node_usage()
    assert usage["node1"]["cpu"] == 1


def test_job_timeout_persisted(db):
    db.create_job("job-timeout", "cpu", None, None, "def f(): pass", timeout_seconds=300)
    job = db.get_job("job-timeout")
    assert job.timeout_seconds == 300


def test_job_timeout_none_by_default(db):
    db.create_job("job-no-timeout", "cpu", None, None, "def f(): pass")
    job = db.get_job("job-no-timeout")
    assert job.timeout_seconds is None


def test_fail_job_timeout_skips_retries(db):
    db.create_job("job-t", "cpu", None, None, "def f(): pass", max_retries=5, timeout_seconds=10)
    db.assign_job("job-t", ["node1"])
    db.fail_job_timeout("job-t", error="Job timed out after 10 seconds")
    job = db.get_job("job-t")
    assert job.status == JobStatus.FAILED
    assert "timed out" in job.error


def test_get_running_jobs(db):
    db.create_job("job-a", "cpu", None, None, "def f(): pass")
    db.create_job("job-b", "cpu", None, None, "def f(): pass")
    db.assign_job("job-a", ["node1"])
    running = db.get_running_jobs()
    assert len(running) == 1
    assert running[0].job_id == "job-a"


def test_get_running_jobs_with_timeout(db):
    db.create_job("job-rt", "cpu", None, None, "def f(): pass", timeout_seconds=60)
    db.assign_job("job-rt", ["node1"])
    running = db.get_running_jobs()
    assert running[0].timeout_seconds == 60


def test_job_depends_on_persisted(db):
    db.create_job("dep1", "cpu", None, None, "def f(): pass")
    db.create_job("job-with-deps", "cpu", None, None, "def f(): pass", depends_on=["dep1"])
    job = db.get_job("job-with-deps")
    assert job.depends_on == ["dep1"]


def test_job_depends_on_none_by_default(db):
    db.create_job("job-no-deps", "cpu", None, None, "def f(): pass")
    job = db.get_job("job-no-deps")
    assert job.depends_on is None


def test_get_pending_jobs_skips_unsatisfied_deps(db):
    db.create_job("dep1", "cpu", None, None, "def f(): pass")
    db.create_job("job-blocked", "cpu", None, None, "def f(): pass", depends_on=["dep1"])
    pending = db.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == "dep1"


def test_get_pending_jobs_includes_satisfied_deps(db):
    db.create_job("dep1", "cpu", None, None, "def f(): pass")
    db.assign_job("dep1", ["node1"])
    db.complete_job("dep1")
    db.create_job("job-ready", "cpu", None, None, "def f(): pass", depends_on=["dep1"])
    pending = db.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].job_id == "job-ready"


def test_cascade_failure_direct(db):
    db.create_job("dep1", "cpu", None, None, "def f(): pass")
    db.create_job("job-child", "cpu", None, None, "def f(): pass", depends_on=["dep1"])
    db.assign_job("dep1", ["node1"])
    db.complete_job("dep1", error="boom")
    db.cascade_failure("dep1")
    child = db.get_job("job-child")
    assert child.status == JobStatus.FAILED
    assert "dep1" in child.error


def test_cascade_failure_recursive(db):
    db.create_job("a", "cpu", None, None, "def f(): pass")
    db.create_job("b", "cpu", None, None, "def f(): pass", depends_on=["a"])
    db.create_job("c", "cpu", None, None, "def f(): pass", depends_on=["b"])
    db.assign_job("a", ["node1"])
    db.complete_job("a", error="boom")
    db.cascade_failure("a")
    assert db.get_job("b").status == JobStatus.FAILED
    assert db.get_job("c").status == JobStatus.FAILED
    assert "a" in db.get_job("b").error
    assert "b" in db.get_job("c").error


def test_cascade_failure_from_cancel(db):
    db.create_job("dep1", "cpu", None, None, "def f(): pass")
    db.create_job("child1", "cpu", None, None, "def f(): pass", depends_on=["dep1"])
    db.assign_job("dep1", ["node1"])
    db.cancel_job("dep1")
    db.cascade_failure("dep1")
    child = db.get_job("child1")
    assert child.status == JobStatus.FAILED
    assert "dep1" in child.error


def test_validate_depends_on_missing(db):
    db.create_job("exists", "cpu", None, None, "def f(): pass")
    missing = db.validate_depends_on(["exists", "does-not-exist"])
    assert missing == ["does-not-exist"]


def test_validate_depends_on_all_valid(db):
    db.create_job("a", "cpu", None, None, "def f(): pass")
    db.create_job("b", "cpu", None, None, "def f(): pass")
    assert db.validate_depends_on(["a", "b"]) == []


def test_cascade_does_not_affect_completed(db):
    db.create_job("dep1", "cpu", None, None, "def f(): pass")
    db.create_job("already-done", "cpu", None, None, "def f(): pass", depends_on=["dep1"])
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
