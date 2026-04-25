from typing import Protocol, TypeVar
from environment.core.jobs import Job, Jobs
from environment.core.machines import Machine, Machines
from enum import IntEnum


class AllocationStatus(IntEnum):
    SUCCESS = 0
    INSUFFICIENT_RESOURCES = 1
    UN_ALLOCATABLE_JOB = 2

J = TypeVar('J', bound=Jobs)
M = TypeVar('M', bound=Machines)

class Allocator(Protocol[M, J]):

    def allocate(self, m: M, j: J) -> AllocationStatus: ...
