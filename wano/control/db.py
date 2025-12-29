import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wano.models.compute import CPUSpec, NodeCapabilities
from wano.models.job import Job, JobStatus


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (node_id TEXT PRIMARY KEY, last_seen TIMESTAMP, status TEXT);
                CREATE TABLE IF NOT EXISTS compute (node_id TEXT, type TEXT, spec_json TEXT, PRIMARY KEY (node_id, type), FOREIGN KEY (node_id) REFERENCES nodes(node_id));
                CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, compute TEXT, gpus INTEGER, status TEXT, node_ids TEXT, created_at TIMESTAMP, started_at TIMESTAMP, completed_at TIMESTAMP, function_code TEXT, error TEXT);
            """)
            conn.commit()

    def _execute(self, query: str, params=()):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, params)
            conn.commit()

    def register_node(self, node_id: str, capabilities: NodeCapabilities):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (node_id, last_seen, status) VALUES (?, ?, ?)",
                (node_id, datetime.now(UTC).isoformat(), "active"),
            )
            for compute_type, spec in capabilities.compute.items():
                if compute_type == "gpu" and isinstance(spec, list):
                    spec_json = json.dumps(
                        [
                            {
                                "name": g.name,
                                "memory_gb": g.memory_gb,
                                "fan_percent": g.fan_percent,
                                "power_usage_w": g.power_usage_w,
                                "power_cap_w": g.power_cap_w,
                                "utilization_percent": g.utilization_percent,
                                "memory_used_mib": g.memory_used_mib,
                            }
                            for g in spec
                        ]
                    )
                elif compute_type == "cpu" and isinstance(spec, CPUSpec):
                    spec_json = json.dumps(
                        {
                            "cores": spec.cores,
                            "memory_gb": spec.memory_gb,
                            "name": spec.name,
                            "temp_celsius": spec.temp_celsius,
                            "power_usage_w": spec.power_usage_w,
                            "power_cap_w": spec.power_cap_w,
                            "utilization_percent": spec.utilization_percent,
                            "memory_used_mib": spec.memory_used_mib,
                        }
                    )
                else:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO compute (node_id, type, spec_json) VALUES (?, ?, ?)",
                    (node_id, compute_type, spec_json),
                )
            conn.commit()

    def update_heartbeat(self, node_id: str):
        self._execute(
            "UPDATE nodes SET last_seen = ? WHERE node_id = ?",
            (datetime.now(UTC).isoformat(), node_id),
        )

    def get_active_nodes(self, heartbeat_timeout_seconds: int = 30) -> list[str]:
        cutoff = datetime.now(UTC) - timedelta(seconds=heartbeat_timeout_seconds)
        cutoff_str = cutoff.isoformat()
        with sqlite3.connect(self.db_path) as conn:
            active = [
                row[0]
                for row in conn.execute(
                    "SELECT node_id FROM nodes WHERE status = 'active' AND last_seen > ?",
                    (cutoff_str,),
                ).fetchall()
            ]
        return active

    def get_available_compute(self, heartbeat_timeout_seconds: int = 30) -> dict[str, list[dict]]:
        cutoff = datetime.now(UTC) - timedelta(seconds=heartbeat_timeout_seconds)
        cutoff_str = cutoff.isoformat()
        result: dict[str, list[dict]] = {}
        with sqlite3.connect(self.db_path) as conn:
            for node_id, compute_type, spec_json in conn.execute(
                "SELECT node_id, type, spec_json FROM compute WHERE node_id IN (SELECT node_id FROM nodes WHERE status = 'active' AND last_seen > ?)",
                (cutoff_str,),
            ).fetchall():
                spec = json.loads(spec_json)
                if isinstance(spec, dict):
                    spec["node_id"] = node_id
                elif isinstance(spec, list):
                    for item in spec:
                        item["node_id"] = node_id
                result.setdefault(compute_type, []).append(spec)
        return result

    def create_job(self, job_id: str, compute: str, gpus: int | None, function_code: str) -> Job:
        now = datetime.now(UTC)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, compute, gpus, status, created_at, function_code) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, compute, gpus, JobStatus.PENDING.value, now.isoformat(), function_code),
            )
            conn.commit()
        return Job(
            job_id=job_id,
            compute=compute,
            gpus=gpus,
            status=JobStatus.PENDING,
            created_at=now,
            function_code=function_code,
        )

    def assign_job(self, job_id: str, node_ids: list[str]):
        self._execute(
            "UPDATE jobs SET node_ids = ?, status = ?, started_at = ? WHERE job_id = ?",
            (json.dumps(node_ids), JobStatus.RUNNING.value, datetime.now(UTC).isoformat(), job_id),
        )

    def complete_job(self, job_id: str, error: str | None = None):
        self._execute(
            "UPDATE jobs SET status = ?, completed_at = ?, error = ? WHERE job_id = ?",
            (
                JobStatus.FAILED.value if error else JobStatus.COMPLETED.value,
                datetime.now(UTC).isoformat(),
                error,
                job_id,
            ),
        )

    def _row_to_job(self, row: tuple) -> Job:
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

    def get_job(self, job_id: str) -> Job | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT job_id, compute, gpus, status, node_ids, created_at, started_at, completed_at, function_code, error FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def get_all_jobs(self) -> list[Job]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT job_id, compute, gpus, status, node_ids, created_at, started_at, completed_at, function_code, error FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_job(row) for row in rows]
