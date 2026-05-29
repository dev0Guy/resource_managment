from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

from environment.core.jobs import JobStatus
from environment.envs import MetricResourceManagementEnvironment


class InvalidMetricActionMasker(gym.Wrapper):

    def __init__(self, env: MetricResourceManagementEnvironment):
        super().__init__(env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def valid_action_mask(self) -> np.ndarray:
        """
        Returns a flat boolean mask of shape (n_machines * n_jobs,).
        A (machine, job) pair is valid only when both machine and job are valid.
        """
        machines = self.env.unwrapped.cluster.machines
        jobs = self.env.unwrapped.cluster.jobs
        allocation_mask = np.zeros((len(machines), len(jobs)), dtype=bool)
        for m_idx, machine in enumerate(machines):
            for j_idx, job in enumerate(jobs):
                if job.status in (JobStatus.Running, JobStatus.Completed, JobStatus.NotCreated):
                    continue

                free_space = machine.capacity - machine.usage
                if np.all(free_space >= job.usage):
                    allocation_mask[m_idx, j_idx] = True

        skip_valid = not allocation_mask.any()
        skip_mask = np.array([skip_valid], dtype=bool)

        return np.concatenate([skip_mask, allocation_mask.flatten()])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)