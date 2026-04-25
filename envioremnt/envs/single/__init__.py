from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from envioremnt.core.allocator import AllocationStatus
from envioremnt.core.cluster import ClusterCreator
from envioremnt.core.jobs import JobStatus
from envioremnt.envs.single.single_action import SingleResourceManagementAction
from envioremnt.envs.single.single_cluster import Jobs, Machines
from envioremnt.envs.single.single_observation import SingleResourceManagementObservation


class SingleResourceManagementEnvironment(gym.Env[SingleResourceManagementObservation, SingleResourceManagementAction]):

    def __init__(self, creator: ClusterCreator[Machines,Jobs]):
        self.creator = creator
        self.cluster = self.creator.create()

        n_machines = len(self.cluster.machines)
        n_jobs = len(self.cluster.jobs)

        self.action_space = gym.spaces.MultiDiscrete([1, n_machines, n_jobs], dtype=np.uint)

        machines_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_machines,),
            dtype=np.uint8
        )
        jobs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_jobs,),
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
            high=1,
            shape=(n_jobs,),
            dtype=np.uint
        )
        length_space = gym.spaces.Box(
            low=0,
            high=1,
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
    ) -> tuple[SingleResourceManagementObservation, dict[str, Any]]:
        self.cluster = self.creator.create()
        return self._get_observation(), {}

    def step(
        self, action: SingleResourceManagementAction
    ) -> tuple[SingleResourceManagementObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        if action.skip:
            return self._skip_tick()

        m_idx, j_idx = action.schedule
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

    def _skip_tick(self) -> tuple[SingleResourceManagementObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        return self._get_observation(), 0, False, False, {}

    def _get_observation(self) -> SingleResourceManagementObservation:
        _machines = np.ndarray([m.usage for m in self.cluster.machines])
        _jobs = np.ndarray([j.usage for j in self.cluster.jobs])
        _status = np.ndarray([j.status for j in self.cluster.jobs])
        _arrival = np.ndarray([j.meta.arrival_time for j in self.cluster.jobs])
        _length = np.ndarray([j.length for j in self.cluster.jobs])
        return SingleResourceManagementObservation(
            machines=_machines,
            jobs=_jobs,
            status=_status,
            arrival=_arrival,
            length=_length
        )