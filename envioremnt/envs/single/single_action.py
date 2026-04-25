from typing import NamedTuple, Tuple


class SingleResourceManagementAction(NamedTuple):
    skip: bool
    schedule: Tuple[int, int]
