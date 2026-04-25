from typing import Protocol


class ClockService(Protocol):

    def tick(self) -> None: ...
