#%% md
### Evaluate Random Model Baseline with W&B
#%%
import glob
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environment.core.jobs import JobStatus
from environment.wrappers.invalid_metric_action_masker import InvalidMetricActionMasker
from environment.wrappers.resource_limit_wrapper import ResourceLimitWrapper
from environment.wrappers.time_limit_penalty import TimeLimitPenaltyWrapper
from environment.wrappers.zero_by_status import ZeroJobUsageByTheirStatus

EPISODES = 100
SEED = 42


N_JOBS = 10
N_MACHINES = 1
N_RESOURCES = 2
N_TICKS = 5
OFFLINE = True
MAX_EPISODE_STEPS = 500
total_timesteps = 500_000

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": total_timesteps,
    "env_name": "resource-management-online",
    "wrappers": ["Monitor", "TimeLimitPenaltyWrapper", "ZeroJobUsageByTheirStatus", "ActionMasker(InvalidMetricActionMasker)"],
    "metadata": {
        "n_jobs": N_JOBS,
        "n_machines": N_MACHINES,
        "n_resources": N_RESOURCES,
        "n_ticks": N_TICKS,
        "is_offline": OFFLINE,
        "max_episode_steps": MAX_EPISODE_STEPS
    },

}


eval_run = wandb.init(
    project="cluster-scheduling",
    job_type="evaluation-random-baseline",
    config=config
)


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
    env = gym.make(
        config["env_name"],
        render_mode="rgb_array",
        n_jobs=n_jobs,
        n_machines=n_machines,
        n_resources=n_resources,
        n_ticks=n_ticks,
        offline=offline,
    )
    # env = ZeroJobUsageByTheirStatus(env, JobStatus.Running, JobStatus.Completed, JobStatus.NotCreated)
    env = ActionMasker(InvalidMetricActionMasker(env), mask_fn)
    env = ResourceLimitWrapper(env)
    env = TimeLimitPenaltyWrapper(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env)
    # MetricRewardWrapper
    return env

import random
import torch

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(SEED)

# --- env ---
vec_env = DummyVecEnv([lambda: make_env(N_JOBS, N_MACHINES, N_RESOURCES, N_TICKS, offline=OFFLINE)])
vec_env = VecVideoRecorder(
    vec_env,
    f"videos/{eval_run.id}",
    record_video_trigger=lambda x: x == 0,
    video_length=10_000,
)

# --- run episodes ---
all_rewards, all_steps = [], []

for ep in range(EPISODES):
    obs = vec_env.reset()
    total_reward, step, done = 0.0, 0, False

    while not done:
        # Sample a valid random action using the action mask
        mask = vec_env.env_method("valid_action_mask")[0]  # shape: (n_actions,)
        valid_actions = np.where(mask)[0]
        action = np.array([np.random.choice(valid_actions)])

        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        step += 1
        done = done[0]
        if step > 300:
            break

    all_rewards.append(total_reward)
    all_steps.append(step)

    wandb.log({
        "eval/episode":  ep + 1,
        "eval/reward":   total_reward,
        "eval/steps":    step,
    })
    print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={step}")

# --- summary ---
wandb.summary["eval/mean_reward"] = sum(all_rewards) / EPISODES
wandb.summary["eval/mean_steps"]  = sum(all_steps)   / EPISODES
wandb.summary["eval/std_reward"]  = float(np.std(all_rewards))
wandb.summary["eval/std_steps"]   = float(np.std(all_steps))

print(f"\navg reward: {sum(all_rewards)/EPISODES:.2f} ± {np.std(all_rewards):.2f} | avg steps: {sum(all_steps)/EPISODES:.1f}")

# --- upload videos ---
vec_env.close()
for f in glob.glob(f"videos/{eval_run.id}/*.mp4"):
    wandb.log({"eval/video": wandb.Video(f, format="mp4")})

wandb.finish()