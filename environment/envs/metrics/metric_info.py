from typing import TypedDict, Optional

from environment.core.allocator import AllocationStatus
from environment.core.jobs import JobStatus


class MetricResourceManagementInformation(TypedDict):
    current_tick: int
    status: list[JobStatus]
    arrival_time: list[int]
    length: list[int]
    allocation: Optional[AllocationStatus]
    schedule_time: list[int]