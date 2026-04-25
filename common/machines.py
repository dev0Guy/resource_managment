from typing import Sequence
from typing import Generic, TypeVar, Iterator, Protocol
from dataclasses import dataclass

Resources = TypeVar('Resources')


@dataclass
class Machine(Generic[Resources]):
    capacity: Resources
    freespace: Resources


Machines = Sequence[Machine[Resources]]
