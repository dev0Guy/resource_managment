import random
from typing import Callable, Generic, TypeVar

import numpy as np
import torch
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv

Observation = TypeVar('Observation')
Action = TypeVar('Action')

class Evaluator(Generic[Observation]):

    def __init__(
        self,
        envs: DummyVecEnv,
        perdict_func: Callable[[DummyVecEnv, Observation], Action]
    ):
        self.perdict_func = perdict_func
        self.envs = envs
        assert envs.num_envs == 1, "Can't work with more than one env."


    def _initlize_before_run(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __call__(self, n_episodes: int = 50, seed: int = 42) -> None:

        for ep in range(n_episodes):
            ep_seed = seed + ep
            self._initlize_before_run(ep_seed)
            self.envs.seed(ep_seed)
            obs = self.envs.reset()
            total_reward, step, done = 0.0, 0, False

            while not done:
                action = self.perdict_func(self.envs, obs)
                obs, reward, done, infos = self.envs.step(action)

                info = infos[0]
                pending_time = []
                for job_idx in range(len(info["arrival_time"])):
                    _time = info["current_tick"] if info["schedule_time"][job_idx] == -1 else info["schedule_time"][job_idx]
                    pending_time.append(_time - info["arrival_time"][job_idx])

                total_reward += float(reward[0])
                step += 1
                done = done[0]

            wandb.log({
                "evaluation/count": ep,
                "evaluation/allocations": sum(1 for x in info["schedule_time"] if x != -1),
                "evaluation/max_pending_time": np.max(pending_time),
                "evaluation/avg_pending_time": np.mean(pending_time),
                "evaluation/tick": info["current_tick"],
                "evaluation/reward": total_reward
            })