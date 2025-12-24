import json
import sqlite3
from datetime import datetime
from pathlib import Path

from wano.models.compute import CPUSpec, NodeCapabilities
from wano.models.job import Job, JobStatus


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (node_id TEXT PRIMARY KEY, last_seen TIMESTAMP, status TEXT);
            CREATE TABLE IF NOT EXISTS compute (node_id TEXT, type TEXT, spec_json TEXT, PRIMARY KEY (node_id, type), FOREIGN KEY (node_id) REFERENCES nodes(node_id));
            CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, compute TEXT, gpus INTEGER, status TEXT, node_ids TEXT, created_at TIMESTAMP, started_at TIMESTAMP, completed_at TIMESTAMP, function_code TEXT, error TEXT);
        """)
        conn.commit()
        conn.close()

    def _execute(self, query: str, params=()):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()

    def register_node(self, node_id: str, capabilities: NodeCapabilities):
        self._execute(
            "INSERT OR REPLACE INTO nodes (node_id, last_seen, status) VALUES (?, ?, ?)",
            (node_id, datetime.utcnow(), "active"),
        )
        for compute_type, spec in capabilities.compute.items():
            if compute_type == "gpu" and isinstance(spec, list):
                spec_json = json.dumps([{"name": g.name, "memory_gb": g.memory_gb} for g in spec])
            elif compute_type == "cpu" and isinstance(spec, CPUSpec):
                spec_json = json.dumps({"cores": spec.cores, "memory_gb": spec.memory_gb})
            else:
                continue
            self._execute(
                "INSERT OR REPLACE INTO compute (node_id, type, spec_json) VALUES (?, ?, ?)",
                (node_id, compute_type, spec_json),
            )

    def update_heartbeat(self, node_id: str):
        self._execute(
            "UPDATE nodes SET last_seen = ? WHERE node_id = ?", (datetime.utcnow(), node_id)
        )

    def get_available_compute(self) -> dict[str, list[dict]]:
        conn = sqlite3.connect(self.db_path)
        result: dict[str, list[dict]] = {}
        for node_id, compute_type, spec_json in conn.execute(
            "SELECT node_id, type, spec_json FROM compute WHERE node_id IN (SELECT node_id FROM nodes WHERE status = 'active')"
        ).fetchall():
            spec = json.loads(spec_json)
            if isinstance(spec, dict):
                spec["node_id"] = node_id
            elif isinstance(spec, list):
                for item in spec:
                    item["node_id"] = node_id
            result.setdefault(compute_type, []).append(spec)
        conn.close()
        return result

    def create_job(self, job_id: str, compute: str, gpus: int | None, function_code: str) -> Job:
        self._execute(
            "INSERT INTO jobs (job_id, compute, gpus, status, created_at, function_code) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, compute, gpus, JobStatus.PENDING.value, datetime.utcnow(), function_code),
        )
        return Job(
            job_id=job_id,
            compute=compute,
            gpus=gpus,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            function_code=function_code,
        )

    def assign_job(self, job_id: str, node_ids: list[str]):
        self._execute(
            "UPDATE jobs SET node_ids = ?, status = ?, started_at = ? WHERE job_id = ?",
            (json.dumps(node_ids), JobStatus.RUNNING.value, datetime.utcnow(), job_id),
        )

    def complete_job(self, job_id: str, error: str | None = None):
        self._execute(
            "UPDATE jobs SET status = ?, completed_at = ?, error = ? WHERE job_id = ?",
            (
                JobStatus.FAILED.value if error else JobStatus.COMPLETED.value,
                datetime.utcnow(),
                error,
                job_id,
            ),
        )

    def get_job(self, job_id: str) -> Job | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT job_id, compute, gpus, status, node_ids, created_at, started_at, completed_at, function_code, error FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return Job(
            job_id=row[0],
            compute=row[1],
            gpus=row[2],
            status=JobStatus(row[3]),
            node_ids=json.loads(row[4]) if row[4] else None,
            created_at=datetime.fromisoformat(row[5]) if row[5] else None,
            started_at=datetime.fromisoformat(row[6]) if row[6] else None,
            completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
            function_code=row[8],
            error=row[9],
        )

    def get_all_jobs(self) -> list[Job]:
        conn = sqlite3.connect(self.db_path)
        jobs = []
        for row in conn.execute(
            "SELECT job_id, compute, gpus, status, node_ids, created_at, started_at, completed_at, function_code, error FROM jobs ORDER BY created_at DESC"
        ).fetchall():
            jobs.append(
                Job(
                    job_id=row[0],
                    compute=row[1],
                    gpus=row[2],
                    status=JobStatus(row[3]),
                    node_ids=json.loads(row[4]) if row[4] else None,
                    created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    started_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    completed_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    function_code=row[8],
                    error=row[9],
                )
            )
        conn.close()
        return jobs
