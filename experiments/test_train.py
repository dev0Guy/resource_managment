import sys
import numpy as np

import stable_baselines3 as sb3

import torch

import sys
import os

from environment.wrappers.resource_limit_wrapper import ResourceLimitWrapper

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

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import logging

logging.basicConfig(level=logging.ERROR)



N_JOBS = 10
N_MACHINES = 1
N_RESOURCES = 2
N_TICKS = 5
OFFLINE = True
MAX_EPISODE_STEPS = 50


total_timesteps = 1_000_000

policy_kwargs = dict(
    net_arch=[128, 64, 128]
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
        "description": """
            Stop run when unscheduled action selected. Make reward to be twice the number of possible rewards. 
            Reward function: 
                skip action -> 0.0
                allocation -> 100 / (1 + current_tick - job_arrival_time)
            Truncated:  step >= 50 or unscheduled action
            State: for each machine get minimum free space, for job takes maximum
        """,
    },

}

run = wandb.init(
    project="cluster-scheduling",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

from environment.wrappers.metric_reward import MetricRewardWrapper
from environment.wrappers.invalid_metric_action_masker import InvalidMetricActionMasker
from sb3_contrib.common.wrappers import ActionMasker


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
    )
    # env = ZeroJobUsageByTheirStatus(env, JobStatus.Running, JobStatus.Completed, JobStatus.NotCreated)
    # env = MetricRewardWrapper(env)
    # env = ActionMasker(InvalidMetricActionMasker(env), mask_fn)
    env = TimeLimitPenaltyWrapper(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = ResourceLimitWrapper(env)
    env = DiscreateFlattenActionWrapper(env)
    env = Monitor(env)
    return env

from sb3_contrib import MaskablePPO # TODO: try with MaskablePPO

env = DummyVecEnv([lambda: make_env(N_JOBS, N_MACHINES, N_RESOURCES, N_TICKS, offline=OFFLINE)])
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2_000 == 0,
    video_length=200,
)


model = PPO(
    config["policy_type"],
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    # changed
    gamma=1,
    gae_lambda=0.99,
    n_epochs=2,
    target_kl=0.02,
)

wandb_callback = WandbCallback(
    gradient_save_freq=1_000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)
metric_callback = CustomMetricsCallback()
from stable_baselines3.common.logger import configure
model.set_logger(configure(f"runs/{run.id}", ["tensorboard", "stdout"]))
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=CallbackList([
        # metric_callback,
        wandb_callback
    ]),
)
model.save(f"models/{run.id}/final_model")
wandb.save(f"models/{run.id}/final_model.zip")
import glob
for f in glob.glob(f"videos/{run.id}/*.mp4"):
    wandb.log({"video": wandb.Video(f, fps=30, format="mp4")})
wandb.finish()


#%% md
### Evaluate Best Model with W&B
#%%
import glob
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = f"models/{run.id}/final_model.zip"
EPISODES = 100

eval_run = wandb.init(
    project="cluster-scheduling",
    job_type="evaluation",
    config={
        "model_path": MODEL_PATH,
        "episodes": EPISODES,
        "n_jobs": N_JOBS,
        "n_machines": N_MACHINES,
        "n_resources": N_RESOURCES,
        "n_ticks": N_TICKS,
        "offline": OFFLINE,
    }
)

# --- env ---
vec_env = DummyVecEnv([lambda: make_env(N_JOBS, N_MACHINES, N_RESOURCES, N_TICKS, offline=OFFLINE)])
vec_env = VecVideoRecorder(
    vec_env,
    f"videos/{eval_run.id}",
    record_video_trigger=lambda x: x == 0,
    video_length=10_000,
)

model = PPO.load(MODEL_PATH, env=vec_env,  policy_kwargs=policy_kwargs)

# --- run episodes ---
all_rewards, all_steps = [], []
# TODO: fix
for ep in range(EPISODES):
    obs = vec_env.reset()
    total_reward, step, done = 0.0, 0, False

    while not done:
        # action_masks = env.valid_action_mask()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        step += 1
        done = done[0]
        if step > 300:
            break

    all_rewards.append(total_reward)
    all_steps.append(step)

    wandb.log({
        "eval/episode":      ep + 1,
        "eval/reward":       total_reward,
        "eval/steps":        step,
    })
    print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={step}")

# --- summary ---
wandb.summary["eval/mean_reward"] = sum(all_rewards) / EPISODES
wandb.summary["eval/mean_steps"]  = sum(all_steps)   / EPISODES
wandb.summary["eval/total_steps"] = sum(all_steps)

print(f"\navg reward: {sum(all_rewards)/EPISODES:.2f} | avg steps: {sum(all_steps)/EPISODES:.1f}")

# --- upload videos (must close env first!) ---
vec_env.close()
for f in glob.glob(f"videos/{eval_run.id}/*.mp4"):
    wandb.log({"eval/video": wandb.Video(f, format="mp4")})

wandb.finish()