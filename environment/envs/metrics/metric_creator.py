import random

import numpy as np

from environment.core.cluster import ClusterCreator
from environment.core.jobs import Metadata, JobStatus, Job
from environment.core.machines import Machine
from environment.envs.single.single_cluster import SingleResourceManagement, Machines, Jobs


class SingleResourceManagementHomogeneousCreator(ClusterCreator):

    def __init__(
        self,
        n_machines: int,
        n_jobs: int,
        n_resources: int,
        n_ticks: int,
        poisson_lambda: float = 5.0,
        offline: bool = True
    ):
        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.n_resources = n_resources
        self.n_ticks = n_ticks
        self.poisson_lambda = poisson_lambda
        self.offline = offline
        self._inner_shape = (n_resources, n_ticks)

    def create(self) -> SingleResourceManagement[Machines, Jobs]:
        return SingleResourceManagement(
            self._create_machines(),
            self._create_jobs()
        )

    def _create_machines(self) -> Machines:
        usage = np.zeros(shape=self._inner_shape)
        capacity = np.zeros(shape=self._inner_shape) + 255
        return [
            Machine(
                capacity=capacity.copy(),
                usage=usage.copy()
            )
            for _ in range(self.n_machines)
        ]

    def _create_jobs(self, long_job_precent: float = 0.2) -> Jobs:
        usage_per_job = np.random.randint(0, 256, size=(self.n_jobs, 1, 1), dtype=np.uint8)
        jobs_usage = np.broadcast_to(
            usage_per_job,
            (self.n_jobs, self.n_resources, self.n_ticks)
        ).copy()

        # one value per job
        jobs_arrival_time = np.random.randint(0, self.n_ticks, size=(self.n_jobs,))
        jobs_length = np.random.randint(0, self.n_ticks, size=(self.n_jobs,))

        # ensure job fits in timeline
        jobs_length = np.minimum(jobs_length, self.n_ticks - jobs_arrival_time)

        # build time axis
        t = np.arange(self.n_ticks)[None, None, :]

        # valid time window mask
        mask = (t >= jobs_arrival_time[:, None, None]) & \
               (t < (jobs_arrival_time + jobs_length)[:, None, None])

        # zero out everything outside the job's active window
        jobs_usage[~mask] = 0

        return [
            Job(
                jobs_length[idx],
                usage=jobs_usage[idx],
                meta=Metadata(arrival_time=jobs_arrival_time[idx]),
                status=JobStatus.Pending if jobs_arrival_time[idx] == 0 else JobStatus.NotCreated
            )
            for idx in range(self.n_jobs)
        ]

