from typing import Sequence
from typing import Generic, TypeVar
from dataclasses import dataclass

Resources = TypeVar('Resources')


@dataclass
class Machine(Generic[Resources]):
    capacity: Resources
    usage: Resources


Machines = Sequence[Machine[Resources]]
