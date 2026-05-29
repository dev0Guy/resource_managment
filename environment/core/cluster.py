from lib2to3.fixes.fix_idioms import TYPE
from typing import Protocol, TypeVar

import numpy as np

from environment.core.jobs import Jobs
from environment.core.machines import Machines
from environment.core.clock import ClockService
from environment.core.allocator import Allocator

J = TypeVar('J', bound=Jobs)
M = TypeVar('M', bound=Machines)

class Cluster(
    Allocator[M, J],
    ClockService,
    Protocol[M, J]
):
    machines: M
    jobs: J
    current_tick: int


class ClusterCreator(Protocol[M, J]):

    def create(self, np_random: np.random.Generator) -> Cluster[M, J]:
        ...
