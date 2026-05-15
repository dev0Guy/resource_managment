from typing import Optional
import logging
from typing import Any, SupportsFloat, Literal

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

from environment.core.allocator import AllocationStatus
from environment.core.cluster import ClusterCreator
from environment.core.jobs import Jobs, JobStatus
from environment.envs.metrics.metric_cluster import Machines
from environment.envs.metrics.metric_info import MetricResourceManagementInformation
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

        self.action_space = gym.spaces.MultiDiscrete(
            [1, n_machines, n_jobs], dtype=np.uint)

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
            "length": length_space,
            "allocation": gym.spaces.Discrete(len(AllocationStatus) + 1)
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MetricResourceManagementObservation, MetricResourceManagementInformation]:
        logger.info("Resting Environment")
        self.cluster = self.creator.create()
        return self._get_observation(None), self.get_information()

    def step(
        self, action: tuple[int, int, int]
    ) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, MetricResourceManagementInformation]:
        action = MetricResourceManagementAction(action[0], action[1:])
        if action.skip:
            return self._skip_tick()

        m_idx, j_idx = action.schedule
        logger.debug("Attempting allocation: machine=%s job=%s", m_idx, j_idx)
        allocation_result = self.cluster.allocate(
            self.cluster.machines[m_idx], self.cluster.jobs[j_idx])

        reward = -1
        are_any_jobs_left = True
        match allocation_result:
            case AllocationStatus.UN_ALLOCATABLE_JOB:
                self._skip_tick()
            case AllocationStatus.INSUFFICIENT_RESOURCES:
                self._skip_tick()
            case AllocationStatus.SUCCESS:
                possible_status_left = {
                    JobStatus.NotCreated, JobStatus.Pending}
                are_all_jobs_completed = all(
                    j.status == JobStatus.Completed for j in self.cluster.jobs)
                are_any_jobs_left = any(
                    j.status in possible_status_left for j in self.cluster.jobs)
                reward = 100 if not are_any_jobs_left else -1
            case _:
                raise RuntimeError("Should be unreachable!")

        return self._get_observation(allocation_result), reward, False, not are_any_jobs_left, self.get_information()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None
        new_info = self.get_information()
        new_observation = self._get_observation(None)
        return self._renderer.render(new_info, new_observation)

    def close(self) -> None:
        pass

    def _get_observation(self, allocation_status: Optional[int]) -> MetricResourceManagementObservation:
        # TODO: check why color of machines stay aloways the same
        _machines = np.array(
            [m.capacity - m.usage for m in self.cluster.machines])
        _jobs = np.array([j.usage for j in self.cluster.jobs])
        _status = np.array([j.status for j in self.cluster.jobs])
        _arrival = np.array([j.meta.arrival_time for j in self.cluster.jobs])
        _length = np.array([j.length for j in self.cluster.jobs])
        allocation_status = len(
            AllocationStatus) if allocation_status is None else allocation_status
        return MetricResourceManagementObservation(
            machines=_machines,
            jobs=_jobs,
            status=_status,
            arrival=_arrival,
            length=_length,
            allocation=allocation_status
        )

    def get_information(self) -> MetricResourceManagementInformation:
        return MetricResourceManagementInformation(
            current_tick=self.cluster.current_tick,
            status=[job.status for job in self.cluster.jobs],
            arrival_time=[job.meta.arrival_time for job in self.cluster.jobs],
            length=[job.length for job in self.cluster.jobs],
        )

    def _skip_tick(self) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, MetricResourceManagementInformation]:
        self.cluster.tick()
        return self._get_observation(None), -1, False, False, self.get_information()

