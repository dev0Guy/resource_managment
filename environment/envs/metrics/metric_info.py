from typing import TypedDict

from environment.core.jobs import JobStatus


class MetricResourceManagementInformation(TypedDict):
    current_tick: int
    status: list[JobStatus]
    arrival_time: list[int]
    length: list[int]