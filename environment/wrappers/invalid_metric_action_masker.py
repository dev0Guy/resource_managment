from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

from environment.core.jobs import JobStatus
from environment.envs import MetricResourceManagementEnvironment


class InvalidMetricActionMasker(gym.Wrapper):

    def __init__(self, env: MetricResourceManagementEnvironment):
        super().__init__(env)
        self.ticks_action = env.action_space[0].n
        self.machine_n = env.action_space[1].n
        self.job_n = env.action_space[2].n
        self.action_space = gym.spaces.Discrete(
            self.job_n * self.machine_n + 1)

    def _decode(self, action):
        if action == 0:
            return np.array([1, 0, 0])

        action -= 1
        machine_idx = action % self.machine_n
        job_idx = action // self.machine_n
        return np.array([0, machine_idx, job_idx], dtype=np.uint32)

    def _encode(self, action):
        if action[0] == 1:
            return 0
        return 1 + int(action[1]) + int(action[2]) * self.machine_n

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        original_action = self._decode(action)
        return self.env.step(original_action)

    def valid_action_mask(self) -> np.ndarray:
        """
        Returns a flat boolean mask of shape (n_machines * n_jobs,).
        A (machine, job) pair is valid only when both machine and job are valid.
        """
        machines = self.env.unwrapped.cluster.machines
        jobs = self.env.unwrapped.cluster.jobs
        allocation_mask = np.zeros((self.machine_n, self.job_n), dtype=bool)
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
