from resource_managment.common.jobs import Jobs
from resource_managment.common.machines import Machines
from resource_managment.common.jobs import JobStatus
from resource_managment.common.clock import Clock
from resource_managment.common.allocator import AllocationStatus, Allocator


class Cluster:

    def __init__(
        self,
        machines: Machines,
        jobs: Jobs,
        allocator: Allocator,
        clock: Clock
    ):
        self._machines = machines
        self._jobs = jobs
        self._allocator = allocator
        self._clock = clock

    def clock_tick(self) -> None:
        self._clock.tick(self._machines, self._jobs)

    def allocate(self, machine_idx: int, job_idx: int) -> AllocationStatus:
        machine = self._machines[machine_idx]
        job = self._jobs[job_idx]
        return self._allocator.allocate(machine, job)

    def has_all_jobs_completed(self) -> bool:
        return all(job.status == JobStatus.COMPLETED for job in self._jobs)

    def has_any_job_pending(self) -> bool:
        return any(job.status == JobStatus.Pending for job in self._jobs)
