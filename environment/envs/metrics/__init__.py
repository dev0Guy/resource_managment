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
            dtype=np.int64
        )
        jobs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_jobs, n_resource, n_ticks),
            dtype=np.int64
        )
        status_space = gym.spaces.Box(
            low=0,
            high=len(JobStatus),
            shape=(n_jobs,),
            dtype=np.int64
        )
        arrival_space = gym.spaces.Box(
            low=0,
            high=n_ticks,
            shape=(n_jobs,),
            dtype=np.int64
        )
        length_space = gym.spaces.Box(
            low=1,
            high=n_ticks,
            shape=(n_jobs,),
            dtype=np.int64
        )
        self._allocation_status = None
        self._reward_on_not_possible_action = -1 * len(self.cluster.jobs) * \
                len(self.cluster.machines) * \
                max(map(lambda j: j.length, self.cluster.jobs))

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
        self._allocation_status = None
        self.cluster = self.creator.create()
        return self._get_observation(None), self.get_information(None)


    def step(
        self, action: tuple[int, int, int]
    ) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, MetricResourceManagementInformation]:
        action = MetricResourceManagementAction(action[0], action[1:])
        m_idx, j_idx = action.schedule

        self._allocation_status = None
        truncated = False
        reward = -1

        if action.skip:
            self._skip_tick()

        else:
            logger.info("Attempting allocation: machine=%s job=%s", m_idx, j_idx)
            self._allocation_status = self.cluster.allocate(self.cluster.machines[m_idx], self.cluster.jobs[j_idx])
            if (
                self._allocation_status == AllocationStatus.UN_ALLOCATABLE_JOB or
                self._allocation_status == AllocationStatus.INSUFFICIENT_RESOURCES
            ):
                self._skip_tick()


        possible_status_left = {
            JobStatus.NotCreated, JobStatus.Pending
        }
        _are_all_jobs_completed = all(
            j.status == JobStatus.Completed for j in self.cluster.jobs
        )
        are_any_jobs_left = any(
            j.status in possible_status_left for j in self.cluster.jobs
        )
        terminated =  not are_any_jobs_left

        if terminated:
            reward = 50_000

        return self._get_observation(self._allocation_status), reward, terminated, truncated, self.get_information(self._allocation_status)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None
        new_info = self.get_information(self._allocation_status)
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
        allocation_status = len(AllocationStatus) if allocation_status is None else allocation_status
        return MetricResourceManagementObservation(
            machines=_machines,
            jobs=_jobs,
            status=_status,
            arrival=_arrival,
            length=_length,
            allocation=allocation_status
        )

    def get_information(self, allocation_status: Optional[AllocationStatus]) -> MetricResourceManagementInformation:
        return MetricResourceManagementInformation(
            current_tick=self.cluster.current_tick,
            status=[job.status for job in self.cluster.jobs],
            arrival_time=[job.meta.arrival_time for job in self.cluster.jobs],
            length=[job.length for job in self.cluster.jobs],
            schedule_time=[job.meta.schedule_time for job in self.cluster.jobs],
            allocation=allocation_status
        )

    def _skip_tick(self) -> tuple[MetricResourceManagementObservation, SupportsFloat, bool, bool, MetricResourceManagementInformation]:
        self.cluster.tick()
        return self._get_observation(None), -1, False, False, self.get_information(None)

