import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np
from typing import TypedDict

from environment.core.jobs import JobStatus
from environment.envs.metrics import MetricResourceManagementObservation


class ResourceLimitObservation(TypedDict):
    jobs: np.ndarray
    machines: np.ndarray


class ResourceLimitWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, pending_status: int = JobStatus.Pending):
        super().__init__(env)
        self.pending_status = pending_status

        orig_space = env.observation_space
        num_jobs = orig_space["jobs"].shape[0]
        numbresources = orig_space["jobs"].shape[1]
        num_machines = orig_space["machines"].shape[0]

        self.observation_space = spaces.Dict({
            "jobs": spaces.Box(
                low=-1,
                high=255,                    # increase if needed
                shape=(num_jobs, numbresources),
                dtype=np.int32               # ← Must be signed!
            ),
            "machines": spaces.Box(
                low=0,
                high=255,
                shape=(num_machines, numbresources),
                dtype=np.int32
            )
        })

    def observation(self, observation: MetricResourceManagementObservation) -> ResourceLimitObservation:
        # Max across resource and time dimensions
        jobs_max = observation["jobs"].max(axis=2)
        machines_max = observation["machines"].min(axis=2)

        jobs_max = jobs_max.astype(np.int32)
        machines_max = machines_max.astype(np.int32)

        pending_mask = (observation["status"] != self.pending_status)
        jobs_max[pending_mask] = -1
        return ResourceLimitObservation(
            jobs=jobs_max,
            machines=machines_max
        )