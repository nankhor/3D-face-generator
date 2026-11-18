"""Background jobs, so long steps (Meshroom, fitting, baking) never block Claude."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    id: str
    name: str
    started: float
    status: str = "running"  # running | done | failed
    result: Any = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    finished: float | None = None

    def write(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self.log.append(line)
        del self.log[:-400]

    def to_dict(self, tail: int = 15) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "seconds": round((self.finished or time.time()) - self.started, 1),
            "error": self.error,
            "log_tail": self.log[-tail:],
            "result": self.result if self.status == "done" else None,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Job:
        job = Job(uuid.uuid4().hex[:8], name, time.time())
        with self._lock:
            self._jobs[job.id] = job

        def runner():
            try:
                job.result = fn(*args, log=job.write, **kwargs)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - reported to the user
                job.error = f"{type(exc).__name__}: {exc}"
                job.write(traceback.format_exc())
                job.status = "failed"
            finally:
                job.finished = time.time()

        threading.Thread(target=runner, name=f"facegen-{name}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def running(self, name: str | None = None) -> list[Job]:
        return [j for j in self._jobs.values() if j.status == "running" and (name is None or j.name == name)]

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started)


JOBS = JobManager()
