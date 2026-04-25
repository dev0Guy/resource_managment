import dataclasses
from typing import Sequence
from typing import Generic, TypeVar, Iterator, Protocol

Resources = TypeVar('Resources')


@dataclasses.dataclass
class Machine(Generic[Resources]):
    capacity: Resources
    usage: Resources


Machines = Sequence[Machine[Resources]]
