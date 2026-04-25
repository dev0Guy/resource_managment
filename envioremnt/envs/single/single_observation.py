from typing import TypedDict, List

import numpy.typing as npt


class SingleResourceManagementObservation(TypedDict):
    machines: npt.NDArray[float]
    jobs: npt.NDArray[float]
    status: npt.NDArray[int]
    arrival: npt.NDArray[int]
    length: npt.NDArray[int]