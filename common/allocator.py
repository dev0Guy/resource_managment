from ast import TypeVar
from typing import Protocol
from resource_managment.common.jobs import Job
from resource_managment.common.machines import Machine
from enum import IntEnum


class AllocationStatus(IntEnum):
    SUCCESS = 0
    INSUFFICIENT_RESOURCES = 1
    UN_ALLOCATABLE_JOB = 2


class AllocatorProtocol(Protocol):

    def allocate(self, m: Machine, j: Job) -> AllocationStatus: ...


Allocator = TypeVar('Allocator', bound=AllocatorProtocol)
