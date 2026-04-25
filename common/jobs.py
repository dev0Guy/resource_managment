from typing import Sequence
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar, Optional
from abc import ABC
from enum import IntEnum, auto
from abc import abstractmethod

Resources = TypeVar('Resources')


class JobStatus(IntEnum):
    NotCreated = auto()
    Pending = auto()
    Running = auto()
    Completed = auto()
    Failed = auto()


@dataclass
class Metadata:
    arrival_time: int
    run_time: int = 0


@dataclass
class Job(Generic[Resources]):
    length: int
    status: JobStatus
    usage: Resources
    meta: Metadata

    def tick_until_complete(self) -> Optional[int]:
        if self.status != JobStatus.Running:
            return None

        return max(0, self.length - self.meta.run_time)


Jobs = Sequence[Job[Resources]]
