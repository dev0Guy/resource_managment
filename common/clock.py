from resource_managment.common.machines import Machines
from resource_managment.common.jobs import Jobs
from typing import Protocol, TypeVar


class ClockProtocol(Protocol):

    def tick(self) -> None: ...


Clock = TypeVar('Clock', bound=ClockProtocol)
