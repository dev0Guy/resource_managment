import logging
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

from environment.core.allocator import AllocationStatus
from environment.core.cluster import ClusterCreator
from environment.core.jobs import Jobs, JobStatus
from environment.envs.metrics.metric_cluster import Machines
from environment.envs.metrics.metric_observation import MetricResourceManagementObservation
from environment.envs.metrics.metrics_action import MetricResourceManagementAction

logger = logging.getLogger(__name__)

class MetricResourceManagementEnvironment(gym.Env[MetricResourceManagementObservation, MetricResourceManagementAction]):

    def __init__(self, creator: ClusterCreator[Machines, Jobs]):
        self.creator = creator
        self.cluster = self.creator.create()

        n_machines = len(self.cluster.machines)
        n_jobs = len(self.cluster.jobs)
        assert self.cluster.machines[0].usage.shape == self.cluster.jobs[0].usage.shape

        n_resource = self.cluster.jobs[0].usage.shape[0]
        n_ticks = self.cluster.jobs[0].usage.shape[1]

        self.action_space = gym.spaces.MultiDiscrete([1, n_machines, n_jobs], dtype=np.uint)

        machines_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_machines, n_resource, n_ticks),
            dtype=np.uint8
        )
        jobs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_jobs, n_resource, n_ticks),
            dtype=np.uint8
        )
        status_space = gym.spaces.Box(
            low=0,
            high=JobStatus.Failed,
            shape=(n_jobs,),
            dtype=np.uint
        )
        arrival_space = gym.spaces.Box(
            low=0,
            high=n_ticks,
            shape=(n_jobs,),
            dtype=np.uint
        )
        length_space = gym.spaces.Box(
            low=1,
            high=n_ticks,
            shape=(n_jobs,),
            dtype=np.uint
        )

        self.observation_space = gym.spaces.Dict({
            "machines": machines_space,
            "jobs": jobs_space,
            "status": status_space,
            "arrival": arrival_space,
            "length": length_space
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MetricResourceManagementObservation, dict[str, Any]]:
        logger.info("Resting Environment")
        self.cluster = self.creator.create()
        return self._get_observation(), {}

    def step(
        self, action: MetricResourceManagementAction
    ) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        if action.skip:
            return self._skip_tick()

        m_idx, j_idx = action.schedule
        logger.debug("Attempting allocation: machine=%s job=%s", m_idx, j_idx)
        allocation_result = self.cluster.allocate(self.cluster.machines[m_idx], self.cluster.jobs[j_idx])

        match allocation_result:
            case AllocationStatus.UN_ALLOCATABLE_JOB:
                return self._skip_tick()
            case AllocationStatus.INSUFFICIENT_RESOURCES:
                return self._skip_tick()
            case AllocationStatus.SUCCESS:
                possible_status_left = {JobStatus.NotCreated, JobStatus.Pending}
                are_all_jobs_completed = all(j.status == JobStatus.Completed for j in self.cluster.jobs)
                are_any_jobs_left = any(j.status in possible_status_left for j in self.cluster.jobs)
                return self._get_observation(), 0, are_all_jobs_completed, not are_any_jobs_left, {}

        raise RuntimeError("Should be unreachable!")

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self) -> None:
        pass

    def _get_observation(self) -> MetricResourceManagementObservation:
        _machines = np.ndarray([m.usage for m in self.cluster.machines])
        _jobs = np.ndarray([j.usage for j in self.cluster.jobs])
        _status = np.ndarray([j.status for j in self.cluster.jobs])
        _arrival = np.ndarray([j.meta.arrival_time for j in self.cluster.jobs])
        _length = np.ndarray([j.length for j in self.cluster.jobs])
        return MetricResourceManagementObservation(
            machines=_machines,
            jobs=_jobs,
            status=_status,
            arrival=_arrival,
            length=_length
        )

    def _skip_tick(self) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        return self._get_observation(), 0, False, False, {}