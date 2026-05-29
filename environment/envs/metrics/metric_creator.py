import random

import numpy as np

from environment.core.cluster import ClusterCreator
from environment.core.jobs import Metadata, JobStatus, Job
from environment.core.machines import Machine
from environment.envs.metrics.metric_cluster import MetricResourceManagement, Machines, Jobs


class MetricResourceManagementHomogeneousCreator(ClusterCreator):

    def __init__(
        self,
        n_machines: int,
        n_jobs: int,
        n_resources: int,
        n_ticks: int,
        offline: bool,
        poisson_lambda: float = 5.0,

    ):
        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.n_resources = n_resources
        self.n_ticks = n_ticks
        self.poisson_lambda = poisson_lambda
        self.offline = offline
        self._inner_shape = (n_resources, n_ticks)

    def create(self, np_random: np.random.Generator) -> MetricResourceManagement:
        return MetricResourceManagement(
            self._create_machines(),
            self._create_jobs(np_random)
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

    def _create_jobs(self, np_random: np.random.Generator, long_job_precent: float = 0.2) -> Jobs:

        jobs_usage = np.zeros(
            (self.n_jobs, self.n_resources, self.n_ticks),
            dtype=np.uint8
        )

        for job_idx in range(self.n_jobs):

            # choose one dominant resource for this job
            dominant_resource = np_random.integers(0, self.n_resources)

            for resource_idx in range(self.n_resources):

                if resource_idx == dominant_resource:
                    # dominant usage: 25% - 50%
                    usage = np_random.integers(64, 129, dtype=np.uint8)

                else:
                    # non-dominant usage: 5% - 10%
                    usage = np_random.integers(13, 26, dtype=np.uint8)

                jobs_usage[job_idx, resource_idx, :] = usage

        # one value per job
        jobs_arrival_time = (
            np.zeros(shape=(self.n_jobs,))
            if self.offline
            else np_random.integers(0, self.n_ticks, size=(self.n_jobs,))
        )

        if self.n_ticks == 1:
            jobs_length = 1
        else:
            n_long = int(self.n_jobs * long_job_precent)
            n_short = self.n_jobs - n_long
            short_jobs = np_random.integers(
                1,
                round(0.06 * self.n_ticks) + 1,
                size=n_short
            )
            long_jobs = np_random.integers(
                round(0.3 * self.n_ticks),
                self.n_ticks + 1,
                size=n_long
            )
            jobs_length = np.concatenate([short_jobs, long_jobs])
            np_random.shuffle(jobs_length)

        # ensure job fits in timeline
        jobs_length = np.minimum(jobs_length, self.n_ticks+1) #- jobs_arrival_time)

        # build time axis (n_jobs, 1, n_ticks)
        t = np.arange(self.n_ticks)[None, None, :]

        # valid time window mask (n_jobs, 1, n_ticks) — broadcasts over resources
        mask = t < jobs_length[:, None, None]
        # mask = (t >= jobs_arrival_time[:, None, None]) & \
        #        (t < (jobs_arrival_time + jobs_length)[:, None, None])

        # zero out everything outside the job's active window
        jobs_usage = np.where(mask, jobs_usage, 0)

        return [
            Job(
                jobs_length[idx],
                usage=jobs_usage[idx],
                meta=Metadata(arrival_time=jobs_arrival_time[idx]),
                status=JobStatus.Pending if jobs_arrival_time[idx] == 0 else JobStatus.NotCreated
            )
            for idx in range(self.n_jobs)
        ]

