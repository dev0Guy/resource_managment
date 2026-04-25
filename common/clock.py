from ast import TypeVar
from resource_managment.common.machines import Machines
from resource_managment.common.jobs import Jobs
from typing import Protocol


class ClockProtocol(Protocol[Jobs, Machines]):

    def tick_machines(self, machines: Machines) -> None: ...

    def tick_jobs(self, jobs: Jobs) -> None: ...


Clock = TypeVar('Clock', bound=ClockProtocol)
