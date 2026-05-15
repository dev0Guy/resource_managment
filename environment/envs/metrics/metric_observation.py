from typing import TypedDict, List

import numpy.typing as npt


class MetricResourceManagementObservation(TypedDict):
    machines: npt.NDArray[int]
    jobs: npt.NDArray[int]
    status: npt.NDArray[int]
    arrival: npt.NDArray[int]
    length: npt.NDArray[int]
    allocation: int
