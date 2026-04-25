from typing import Protocol, TypeVar
from resource_managment.common.jobs import Job, Jobs
from resource_managment.common.machines import Machine, Machines
from enum import IntEnum


class AllocationStatus(IntEnum):
    SUCCESS = 0
    INSUFFICIENT_RESOURCES = 1
    UN_ALLOCATABLE_JOB = 2


class AllocatorProtocol(Protocol[Jobs, Machines]):

    def allocate(self, m: Machine, j: Job) -> AllocationStatus: ...


Allocator = TypeVar('Allocator', bound=(AllocatorProtocol,))
