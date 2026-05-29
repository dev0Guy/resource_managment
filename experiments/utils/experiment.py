from typing import Type, Callable

import gymnasium
import wandb
from jedi.inference.gradual.typing import TypedDict
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
import glob

from environment.wrappers.action_flattener import DiscreateFlattenActionWrapper
from environment.wrappers.time_limit_penalty import TimeLimitPenaltyWrapper
from experiments.utils.evaluator import Evaluator
from experiments.utils.wandb_custome_metric import CustomMetricsCallback


class ExperimentConfig(TypedDict):
    max_episode_steps: int
    number_of_jobs: int
    number_of_machines: int
    number_of_resources: int
    number_of_ticks: int
    is_offline: bool
    total_timesteps: int
    algorithm_class: Type[BaseAlgorithm]
    algorithm_kwargs: dict
    policy_kwargs: dict
    policy_type: str
    wrapper_func: Callable[[gymnasium.Env], gymnasium.Env]

class ExperimentRunner:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run = wandb.init(
            project="cluster-scheduling",
            sync_tensorboard=True,
            config={
                "policy_type": self.config["policy_type"]
            },
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({
            "Reward Function": f"""
                deepRM reward function
                    skip action -> sum(1/length for each pending or running jobs)
                    allocation -> 0.0
                Truncated:  step >= {self.config["max_episode_steps"]}  -> 0.0
                Success: 0.0
            """,
            "State": "base dict state",
            "Algorithm": self.config["algorithm_class"],
            "Limit": self.config["max_episode_steps"],
            "PolicyArguments": self.config["policy_kwargs"],
            "description": "fixed skip reward"
        })


    def create_env(self) -> gymnasium.Env:
        env = gymnasium.make(
            'resource-management-online',
            render_mode="rgb_array",
            n_jobs=self.config["number_of_jobs"],
            n_machines=self.config["number_of_machines"],
            n_resources=self.config["number_of_resources"],
            n_ticks=self.config["number_of_ticks"],
            offline=self.config["is_offline"],
            disable_env_checker=True
        )
        env = TimeLimitPenaltyWrapper(env, max_episode_steps=self.config["max_episode_steps"])
        env = DiscreateFlattenActionWrapper(env)
        env = Monitor(env)
        return self.config["wrapper_func"](env)

    def callbacks(self) -> CallbackList:
        return CallbackList([
            WandbCallback(
                gradient_save_freq=1_000,
                model_save_path=f"models/{self.run.id}",
                verbose=2,
            ),
            CustomMetricsCallback()
        ])

    def _run_training(self) -> None:
        envs = DummyVecEnv([lambda: self.create_env()])
        envs = VecVideoRecorder(
            envs,
            f"videos/{self.run.id}",
            record_video_trigger=lambda x: x % 2_000 == 0,
            video_length=200,
        )
        algorithm = self.config["algorithm_class"](
            policy=self.config["policy_type"],
            env=envs,
            verbose=1,
            tensorboard_log=f"runs/{self.run.id}",
            policy_kwargs=self.config["policy_kwargs"],
            **self.config["algorithm_kwargs"]
        )
        algorithm.set_logger(configure(f"runs/{self.run.id}", ["tensorboard", "stdout"]))
        algorithm.learn(self.config["total_timesteps"], callback=self.callbacks())
        algorithm.save(f"models/{self.run.id}/final_model")
        self.model_path = f"models/{self.run.id}/final_model.zip"
        wandb.save(self.model_path)
        for f in glob.glob(f"videos/{self.run.id}/*.mp4"):
            wandb.log({"video": wandb.Video(f, fps=30, format="mp4")})

    def _run_evaluation(self) -> None:
        envs = DummyVecEnv([lambda: self.create_env()])
        envs = VecVideoRecorder(
            envs,
            f"videos/evaluation/{self.run.id}",
            record_video_trigger=lambda x: x % 2_000 == 0,
            video_length=200,
        )
        algorithm = self.config["algorithm_class"].load(self.model_path,env=envs, policy_kwargs=self.config["policy_kwargs"])
        self.evaluator = Evaluator(envs, lambda env, obs: algorithm.predict(obs, deterministic=True)[0])
        self.evaluator()
        envs.close()
        for f in glob.glob(f"videos/evaluation/{self.run.id}/*.mp4"):
            wandb.log({"evaluation/video": wandb.Video(f, format="mp4")})
        wandb.finish()


    def run_experiment(self) -> None:
        self._run_training()
        self._run_evaluation()



