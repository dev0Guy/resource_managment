from typing import Protocol, TypeVar
from envioremnt.core.jobs import Job, Jobs
from envioremnt.core.machines import Machine, Machines
from enum import IntEnum


class AllocationStatus(IntEnum):
    SUCCESS = 0
    INSUFFICIENT_RESOURCES = 1
    UN_ALLOCATABLE_JOB = 2


class Allocator(Protocol[Jobs, Machines]):

    def allocate(self, m: Machine, j: Job) -> AllocationStatus: ...
