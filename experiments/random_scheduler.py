import glob
import sys
import numpy as np

import stable_baselines3 as sb3

import torch

import sys
import os

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.dqn.policies import DQNPolicy

from environment.envs.metrics import MetricResourceManagementInformation
from environment.wrappers.invalid_metric_action_masker import InvalidMetricActionMasker
from environment.wrappers.resource_limit_wrapper import ResourceLimitWrapper
from experiments.utils.evaluator import Evaluator

sys.path.append(os.path.abspath(".."))
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from environment.wrappers.time_limit_penalty import TimeLimitPenaltyWrapper
from environment.wrappers.action_flattener import FlattenActionWrapper, DiscreateFlattenActionWrapper
from environment.wrappers.zero_by_status import ZeroJobUsageByTheirStatus
from environment.core.jobs import JobStatus
from experiments.utils.wandb_custome_metric import CustomMetricsCallback, EpisodesMetrics
from environment.envs import *


from stable_baselines3.common.callbacks import CallbackList
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import logging

logging.basicConfig(level=logging.ERROR)



N_JOBS = 10
N_MACHINES = 1
N_RESOURCES = 3
N_TICKS = 10
OFFLINE = True
MAX_EPISODE_STEPS = N_JOBS * (0.4 * N_TICKS) + N_TICKS
EPISODES = 100
SEED = 42

total_timesteps = 500_000
# total_timesteps = 500_000

policy_kwargs = dict(
    net_arch=[64, 32, 32]
)

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": total_timesteps,
    "env_name": "resource-management-online",
    "wrappers": [
        "TimeLimitPenaltyWrapper",
        "ResourceLimitWrapper",
        "DiscreateFlattenActionWrapper",
        "Monitor",
    ],
    "metadata": {
        "n_jobs": N_JOBS,
        "n_machines": N_MACHINES,
        "n_resources": N_RESOURCES,
        "n_ticks": N_TICKS,
        "is_offline": OFFLINE,
        "max_episode_steps": MAX_EPISODE_STEPS,
    },

}

run = wandb.init(
    project="cluster-scheduling",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

model_config = {
    "gamma": 1,
    # "n_steps": 5_000,
    # 'batch_size': 256
}

wandb.config.update({
    "Reward Function": f"""
        deepRM reward function
            skip action -> sum(1/length for each pending or running jobs)
            allocation -> 0.0
        Truncated:  step >= {MAX_EPISODE_STEPS}  -> 0.0
        Success: 0.0
    """,
    "Model": model_config,
    "State": "base dict state",
    "Algorithm": "DQN",
    "Limit": MAX_EPISODE_STEPS,
    "PolicyArguments": policy_kwargs,
    "description": "Random scheduler, fixed skip reward"

})

def mask_fn(env) -> np.ndarray:
    return env.valid_action_mask()


def make_env(
        n_jobs: int,
        n_machines: int,
        n_resources: int,
        n_ticks: int,
        max_episode_steps: int = 300,
        offline: bool = True,
):
    print(f"{n_jobs=}, {n_machines=}, {n_resources=}, {n_ticks=}, {max_episode_steps=}")
    env: MetricResourceManagementEnvironment = gym.make( # type: ignore
        config["env_name"],
        render_mode="rgb_array",
        n_jobs=n_jobs,
        n_machines=n_machines,
        n_resources=n_resources,
        n_ticks=n_ticks,
        offline=offline,
        autoreset=False,
        disable_env_checker=True
    )
    env = TimeLimitPenaltyWrapper(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = DiscreateFlattenActionWrapper(env)
    env = ActionMasker(InvalidMetricActionMasker(env), mask_fn)
    env = Monitor(env)
    return env


envs = DummyVecEnv([lambda: make_env(N_JOBS, N_MACHINES, N_RESOURCES, N_TICKS, offline=OFFLINE)])
envs = VecVideoRecorder(
    envs,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2_000 == 0,
    video_length=200,
)

def predict_func(env, obs) -> None:
    mask = env.env_method("valid_action_mask")[0]  # shape: (n_actions,)
    valid_actions = np.where(mask)[0]
    return np.array([np.random.choice(valid_actions)])

evaluator = Evaluator(envs, predict_func)
evaluator()
for f in glob.glob(f"videos/{run.id}/*.mp4"):
    wandb.log({"evaluation/video": wandb.Video(f, format="mp4")})
wandb.finish()
