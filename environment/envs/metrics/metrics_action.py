from typing import NamedTuple, Tuple


class MetricResourceManagementAction(NamedTuple):
    skip: bool
    schedule: Tuple[int, int]
