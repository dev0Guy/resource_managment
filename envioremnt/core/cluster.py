from lib2to3.fixes.fix_idioms import TYPE
from typing import Protocol, TypeVar

from envioremnt.core.jobs import Jobs
from envioremnt.core.machines import Machines
from envioremnt.core.clock import ClockService
from envioremnt.core.allocator import Allocator

ClusterCreationParameters = TypeVar('ClusterCreationParameters')

class Cluster(
    Allocator[Machines, Jobs],
    ClockService,
    Protocol[Machines, Jobs]
):
    machines: Machines
    jobs: Jobs
    current_tick: int


class ClusterCreator(Protocol[Machines, Jobs]):

    def create(self) -> Cluster[Machines, Jobs]:
        ...
