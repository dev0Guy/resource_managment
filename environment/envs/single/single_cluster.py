import logging
from typing import List

from environment.core.allocator import AllocationStatus
from environment.core.cluster import Cluster
from environment.core.jobs import Job, JobStatus
from environment.core.machines import Machine

Jobs = List[Job[int]]
Machines = List[Machine[int]]

logger = logging.getLogger(__name__)

class SingleResourceManagement(Cluster[Machines, Jobs]):

    def __init__(self, machines: Machines, jobs: Jobs):
        self.machines = machines
        self.jobs = jobs
        self.current_tick = 0
        logger.info("Cluster initialized with %d machines and %d jobs",len(machines), len(jobs))

    def allocate(self, m: Machine, j: Job) -> AllocationStatus:
        if j.status is not JobStatus.Pending:
            logger.warning("Job not pending: %s", j.status)
            return AllocationStatus.UN_ALLOCATABLE_JOB

        free_space = m.capacity - m.usage

        if j.usage > free_space:
            logger.warning("Insufficient resources")
            return AllocationStatus.INSUFFICIENT_RESOURCES

        m.usage += j.usage
        j.status = JobStatus.Running
        j.meta.run_time = 0

        return AllocationStatus.SUCCESS

    def tick(self) -> None:
        logger.debug("Tick %d", self.current_tick)
        self.current_tick += 1
        self._jobs_tick()
        self._machines_tick()

    def _jobs_tick(self) -> None:
        for j in self.jobs:
            match j.status:
                case JobStatus.Running:
                    logger.debug("Running job tick: %s", j)
                    j.meta.run_time += 1
                    j.status = JobStatus.Completed
                    j.usage = 0
                case JobStatus.NotCreated if j.meta.arrival_time == self.current_tick:
                    logger.debug("Job arrived: %s", j)
                    j.meta.run_time = 0
                    j.status = JobStatus.Pending
                case _:
                    pass

    def _machines_tick(self) -> None:
        for idx, m in enumerate(self.machines):
            logger.debug("Resetting machine[%d] usage", idx)
            m.usage = 0
