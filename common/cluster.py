from lib2to3.fixes.fix_idioms import TYPE
from typing import Protocol, TypeVar

from resource_managment.common.jobs import Jobs
from resource_managment.common.machines import Machines
from resource_managment.common.clock import ClockProtocol
from resource_managment.common.allocator import AllocatorProtocol

ClusterCreationParameters = TypeVar('ClusterCreationParameters')

class ClusterProtocol(
    AllocatorProtocol[Machines, Jobs],
    ClockProtocol,
    Protocol[Machines, Jobs]
):
    machines: Machines
    jobs: Jobs
    current_tick: int

Cluster = TypeVar('Cluster', bound=(ClusterProtocol,))


class ClusterCreatorProtocol(Protocol[Machines, Jobs]):

    def create(self) -> ClusterProtocol[Machines, Jobs]:
        ...


ClusterCreator = TypeVar('ClusterCreator', bound=ClusterCreatorProtocol)
