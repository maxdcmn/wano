from wano.control.scheduler import Scheduler
from wano.models.job import Job, JobStatus


def test_cpu_job_with_available_compute():
    scheduler = Scheduler()
    job = Job(job_id="job1", compute="cpu", status=JobStatus.PENDING)
    available = {"cpu": [{"node_id": "node1", "cores": 4}]}

    assert scheduler.schedule_job(job, available) == ["node1"]


def test_cpu_job_without_available_compute():
    scheduler = Scheduler()
    job = Job(job_id="job1", compute="cpu", status=JobStatus.PENDING)

    assert scheduler.schedule_job(job, {}) is None


def test_gpu_job_single_node():
    scheduler = Scheduler()
    job = Job(job_id="job1", compute="gpu", gpus=1, status=JobStatus.PENDING)
    available = {"gpu": [[{"node_id": "node1"}]]}

    assert scheduler.schedule_job(job, available) == ["node1"]


def test_gpu_job_multi_node():
    scheduler = Scheduler()
    job = Job(job_id="job2", compute="gpu", gpus=3, status=JobStatus.PENDING)
    available = {"gpu": [[{"node_id": "node1"}], [{"node_id": "node1"}], [{"node_id": "node2"}]]}

    assert scheduler.schedule_job(job, available) == ["node1", "node1", "node2"]


def test_gpu_job_insufficient_gpus():
    scheduler = Scheduler()
    job = Job(job_id="job3", compute="gpu", gpus=5, status=JobStatus.PENDING)
    available = {"gpu": [[{"node_id": "node1"}], [{"node_id": "node2"}]]}

    assert scheduler.schedule_job(job, available) is None


def test_gpu_job_defaults_to_one():
    scheduler = Scheduler()
    job = Job(job_id="job4", compute="gpu", gpus=None, status=JobStatus.PENDING)
    available = {"gpu": [[{"node_id": "node1"}]]}

    assert scheduler.schedule_job(job, available) == ["node1"]


def test_cpu_job_respects_usage():
    scheduler = Scheduler()
    job = Job(job_id="job5", compute="cpu", status=JobStatus.PENDING)
    available = {
        "cpu": [
            {"node_id": "node1", "cores": 2},
            {"node_id": "node2", "cores": 1},
        ]
    }
    usage = {"node1": {"cpu": 2, "gpu": 0}, "node2": {"cpu": 0, "gpu": 0}}

    assert scheduler.schedule_job(job, available, usage) == ["node2"]


def test_gpu_job_respects_usage():
    scheduler = Scheduler()
    job = Job(job_id="job6", compute="gpu", gpus=1, status=JobStatus.PENDING)
    available = {"gpu": [[{"node_id": "node1"}, {"node_id": "node1"}], [{"node_id": "node2"}]]}
    usage = {"node1": {"cpu": 0, "gpu": 2}, "node2": {"cpu": 0, "gpu": 0}}

    assert scheduler.schedule_job(job, available, usage) == ["node2"]
