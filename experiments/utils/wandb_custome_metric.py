import pprint

from environment.core.allocator import AllocationStatus
from environment.core.jobs import JobStatus
from collections import defaultdict
from typing import TypedDict, List
from gymnasium.wrappers import TimeLimit

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from environment.envs.metrics.metric_info import MetricResourceManagementInformation


class EpisodesMetrics(TypedDict):
    count: int
    length: List[int]
    reward: List[float]
    avg_pending_time: List[float]
    max_pending_time: List[float]
    current_tick: List[int]
    allocations: List[int]
    failed_allocations: List[int]
    failed_allocations_usage: List[int]
    failed_allocations_status: List[int]



class CustomMetricsCallback(BaseCallback):
    """
    Pulls custom keys from the Monitor's episode_info buffer
    and logs them to W&B at every rollout end.
    """
    # TODO: make this work using only info and not the state itself

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_metrics = EpisodesMetrics(
            count=0,
            allocations=[],
            length=[],
            reward=[],
            avg_pending_time=[],
            max_pending_time=[],
            current_tick=[],
            failed_allocations=[],
            failed_allocations_usage=[],
            failed_allocations_status=[]
        )

    def _update_episode_metrics(self):
        for key in self.episode_metrics:
            if key == "count": continue
            self.episode_metrics[key].clear()



    def _on_step(self) -> bool:
        # VecEnv returns arrays
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for done, info in zip(dones, infos):
            if not done:
                failed_due_status = info["allocation"] == AllocationStatus.UN_ALLOCATABLE_JOB
                failed_due_usage = info["allocation"] == AllocationStatus.INSUFFICIENT_RESOURCES
                self.episode_metrics["failed_allocations"].append(failed_due_usage + failed_due_status)
                self.episode_metrics["failed_allocations_usage"].append(failed_due_usage)
                self.episode_metrics["failed_allocations_status"].append(failed_due_status)
                continue

            if "episode" in info:
                self.episode_metrics["reward"].append(info["episode"]["r"])
                self.episode_metrics["length"].append(info["episode"]["l"])

            pending_time = []
            for job_idx in range(len(info["arrival_time"])):
                _time = info["current_tick"] if info["schedule_time"][job_idx] == -1 else info["schedule_time"][job_idx]
                pending_time.append(_time - info["arrival_time"][job_idx])

            self.episode_metrics["avg_pending_time"].append(np.average(pending_time))
            self.episode_metrics["max_pending_time"].append(np.max(pending_time))
            self.episode_metrics["current_tick"].append(info["current_tick"])
            self.episode_metrics["allocations"].append(sum(1 for x in info["schedule_time"] if x != -1))

        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return

        for idx in range(len(self.episode_metrics["length"])):
            self.episode_metrics["count"] += 1
            wandb.log({
                "episode/count": self.episode_metrics["count"],
                "episode/length": self.episode_metrics["length"][idx],
                "episode/reward": self.episode_metrics["reward"][idx],
                "episode/allocations": self.episode_metrics["allocations"][idx],
                "episode/max_pending_time": self.episode_metrics["max_pending_time"][idx],
                "episode/avg_pending_time": self.episode_metrics["avg_pending_time"][idx],
                "episode/tick": self.episode_metrics["current_tick"][idx],
                "episode/failed_allocations": np.sum(self.episode_metrics["failed_allocations"][idx]),
                "episode/failed_allocations_usage": np.sum(self.episode_metrics["failed_allocations_usage"][idx]),
                "episode/failed_allocations_status": np.sum(self.episode_metrics["failed_allocations_status"][idx]),
            })

        self._update_episode_metrics()



    # def _on_rollout_end(self) -> None:
    #     if len(self.model.ep_info_buffer) == 0:
    #         return
    #
    #     wandb.log({
    #         "scheduling/avg_pending_time": np.mean(self._episode_metrics["avg_pending_time"]),
    #         "scheduling/avg_turnaround_time": np.mean(self._episode_metrics["avg_turnaround_time"]),
    #         "scheduling/avg_utilization": np.mean(self._episode_metrics["avg_utilization"]),
    #         "scheduling/avg_imbalance": np.mean(self._episode_metrics["avg_imbalance"]),
    #         "global_step": self.num_timesteps,
    #     })
    #     self._episode_metrics.clear()
    #
    # def _on_step(self) -> bool:
    #
    #     info: MetricResourceManagementInformation
    #     for info in self.locals["infos"]:
    #         job_statuses = info["status"]
    #         current_tick = info["current_tick"]
    #         for job_id, status in enumerate(job_statuses):
    #             if status == JobStatus.Pending:
    #                 self._pending_ticks[job_id] += 1
    #                 if job_id not in self._first_pending:
    #                     self._first_pending[job_id] = current_tick
    #                     self._turnaround_time[job_id] = current_tick - \
    #                         self._first_pending[job_id]
    #             if status == JobStatus.Completed:
    #                 self._turnaround_time[job_id] = current_tick - \
    #                     self._first_pending[job_id]
    #
    #         if "episode" not in info:
    #             continue
    #     #
    #         if self._pending_ticks:
    #             avg_pending_time = np.mean(list(self._pending_ticks.values()))
    #             turnaround_time = np.mean(list(self._turnaround_time.values()))
    #             self._episode_metrics["avg_pending_time"].append(
    #                 avg_pending_time)
    #             self._episode_metrics["avg_turnaround_time"].append(
    #                 turnaround_time)
    #             self._episode_metrics["avg_utilization"].append(
    #                 (1 - self.locals["obs_tensor"]["machines"].mean()))
    #             self._episode_metrics["avg_imbalance"].append(
    #                 self.locals["obs_tensor"]["machines"].std())
    #         self._pending_ticks.clear()
    #     return True
