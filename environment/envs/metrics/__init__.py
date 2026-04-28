import logging
from typing import Any, SupportsFloat, Literal

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

from environment.core.allocator import AllocationStatus
from environment.core.cluster import ClusterCreator
from environment.core.jobs import Jobs, JobStatus
from environment.envs.metrics.metric_cluster import Machines
from environment.envs.metrics.metric_observation import MetricResourceManagementObservation
from environment.envs.metrics.metric_renderer import ClusterMetricRenderer
from environment.envs.metrics.metrics_action import MetricResourceManagementAction

logger = logging.getLogger(__name__)

class MetricResourceManagementEnvironment(gym.Env[MetricResourceManagementObservation, MetricResourceManagementAction]):

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    # TODO: add render function
    def __init__(
        self,
        creator: ClusterCreator[Machines, Jobs],
        render_mode: Literal['human', 'rgb_array', None] = None,
    ):
        self.creator = creator
        self.cluster = self.creator.create()
        self.render_mode = render_mode

        self._renderer = ClusterMetricRenderer(self.render_mode)

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
            dtype=np.float32
        )
        jobs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_jobs, n_resource, n_ticks),
            dtype=np.float32
        )
        status_space = gym.spaces.Box(
            low=0,
            high=len(JobStatus),
            shape=(n_jobs,),
            dtype=np.float32
        )
        arrival_space = gym.spaces.Box(
            low=0,
            high=n_ticks,
            shape=(n_jobs,),
            dtype=np.float32
        )
        length_space = gym.spaces.Box(
            low=1,
            high=n_ticks,
            shape=(n_jobs,),
            dtype=np.float32
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
        self, action: tuple[int, int, int]
    ) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        action = MetricResourceManagementAction(action[0], action[1:])
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
                reward = 100 if not are_any_jobs_left else -1
                return self._get_observation(), reward, False, not are_any_jobs_left, {}

        raise RuntimeError("Should be unreachable!")

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None
        new_info = {"current_tick": self.cluster.current_tick}
        new_observation = self._get_observation()
        return self._renderer.render(new_info, new_observation)

    def close(self) -> None:
        pass

    def _get_observation(self) -> MetricResourceManagementObservation:
        # TODO: check why color of machines stay aloways the same
        _machines = np.array([m.capacity - m.usage for m in self.cluster.machines])
        _jobs = np.array([j.usage for j in self.cluster.jobs])
        _status = np.array([j.status for j in self.cluster.jobs])
        _arrival = np.array([j.meta.arrival_time for j in self.cluster.jobs])
        _length = np.array([j.length for j in self.cluster.jobs])
        return MetricResourceManagementObservation(
            machines=_machines,
            jobs=_jobs,
            status=_status,
            arrival=_arrival,
            length=_length
        )

    def _skip_tick(self) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        self.cluster.tick()
        return self._get_observation(), -1, False, False, {}