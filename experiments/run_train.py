import logging
import sys

import gymnasium
import numpy as np

import stable_baselines3 as sb3

import torch

import sys
import os

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy

from environment.envs.metrics import MetricResourceManagementInformation
from environment.wrappers.resource_limit_wrapper import ResourceLimitWrapper
from experiments.utils.experiment import ExperimentRunner, ExperimentConfig

sys.path.append(os.path.abspath(".."))
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from environment.wrappers.time_limit_penalty import TimeLimitPenaltyWrapper
from environment.wrappers.action_flattener import FlattenActionWrapper, DiscreateFlattenActionWrapper
from environment.wrappers.zero_by_status import ZeroJobUsageByTheirStatus
from environment.core.jobs import JobStatus
from experiments.utils.wandb_custome_metric import CustomMetricsCallback
from environment.envs import *


from stable_baselines3.common.callbacks import CallbackList
import gymnasium as gym
N_JOBS = 10
N_MACHINES = 1
N_RESOURCES = 3
N_TICKS = 10
OFFLINE = True
MAX_EPISODE_STEPS = N_JOBS * (0.4 * N_TICKS) + N_TICKS
TOTAL_TIMESTEPS = 1_000_000

def environment_wrapper(env: gymnasium.Env) -> gymnasium.Env:
    return ResourceLimitWrapper(env)

if __name__ == '__main__':

    logging.basicConfig(level=logging.ERROR)
    config = dict(
        max_episode_steps=MAX_EPISODE_STEPS,
        number_of_jobs=N_JOBS,
        number_of_machines=N_MACHINES,
        number_of_resources=N_RESOURCES,
        number_of_ticks=N_TICKS,
        is_offline=OFFLINE,
        total_timesteps=TOTAL_TIMESTEPS,
        algorithm_class=DQN,
        algorithm_kwargs={},
        policy_kwargs=dict(
            net_arch=[64, 32, 32]
        ),
        policy_type="MultiInputPolicy",
        wrapper_func=environment_wrapper#lambda env: env
    )
    ExperimentRunner(config).run_experiment()