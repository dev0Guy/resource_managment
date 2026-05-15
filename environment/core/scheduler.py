from abc import ABC
from typing import Generic, TypeVar, NamedTuple, Tuple

Machines = TypeVar('Machines')
Jobs = TypeVar('Jobs')

class SchedulerAction(NamedTuple):
    skip: bool
    schedule: Tuple[int, int]

class AbstractScheduler(ABC, Generic[Machines, Jobs]):

    def schedule(self) -> SchedulerAction: ...

class RoundRobinScheduler(AbstractScheduler[Machines, Jobs]):

    def schedule(self) -> SchedulerAction:
        raise NotImplemented

class RandomScheduler(AbstractScheduler[Machines, Jobs]):

    def schedule(self) -> SchedulerAction:
        raise NotImplemented

class TetrisScheduler(AbstractScheduler[Machines, Jobs]):
    def schedule(self) -> SchedulerAction:
        raise NotImplemented

class PackerScheduler(AbstractScheduler[Machines, Jobs]):
    def schedule(self) -> SchedulerAction:
        raise NotImplemented