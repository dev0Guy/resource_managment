import logging
from typing import List

import numpy as np

from environment.core.allocator import AllocationStatus
from environment.core.cluster import Cluster
from environment.core.jobs import Job, JobStatus
from environment.core.machines import Machine
import numpy.typing as npt

Jobs = List[Job[npt.NDArray[int]]]
Machines = List[Machine[npt.NDArray[int]]]

logger = logging.getLogger(__name__)


class MetricResourceManagement(Cluster[Machines, Jobs]):

    def __init__(self, machines: Machines, jobs: Jobs):
        self.machines = machines
        self.jobs = jobs
        self.current_tick = 0

    def allocate(self, m: Machine[npt.NDArray[int]], j: Job[npt.NDArray[int]]) -> AllocationStatus:
        if j.status is not JobStatus.Pending:
            logger.warning("Job not pending: %s", j.status)
            return AllocationStatus.UN_ALLOCATABLE_JOB

        free_space = m.capacity - m.usage

        if np.any(j.usage > free_space):
            logger.warning("Insufficient resources")
            return AllocationStatus.INSUFFICIENT_RESOURCES

        m.usage += j.usage
        j.status = JobStatus.Running
        j.meta.run_time = 0
        j.meta.schedule_time = self.current_tick

        return AllocationStatus.SUCCESS

    def tick(self) -> None:
        logger.debug("Tick %d", self.current_tick)
        self.current_tick += 1
        self._jobs_tick()
        self._machines_tick()

    def _jobs_tick(self) -> None:
        for idx, j in enumerate(self.jobs):
            match j.status:
                case JobStatus.Running if j.meta.run_time == j.length:
                    logger.debug("Change from Running to Completed job[%s]", idx)
                    j.status = JobStatus.Completed
                case JobStatus.Running:
                    logger.debug("Tick for running job [%s]", idx)
                    j.meta.run_time += 1
                case JobStatus.NotCreated if j.meta.arrival_time == self.current_tick:
                    logger.debug("Job arrived: %s", j)
                    j.meta.run_time = 0
                    j.status = JobStatus.Pending
                case _:
                    pass

    def _machines_tick(self) -> None:
        for idx, m in enumerate(self.machines):
            logger.debug("Resetting machine[%d] usage", idx)
            m.usage[:, :-1] = m.usage[:, 1:]
            m.usage[:, -1] = 0

