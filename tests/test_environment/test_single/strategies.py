from hypothesis import strategies as st

from environment.core.jobs import Job, Metadata, JobStatus
from environment.core.machines import Machine


__all__ = [
    'machines_st',
    'jobs_st'
]


@st.composite
def machine_st(
    draw
) -> Machine[int]:
    capacity = 255
    usage = draw(st.integers(0, capacity))
    return Machine(capacity, usage)


@st.composite
def machines_st(
    draw,
    min_size: int = 1,
    max_size: int = 5,
) -> list[Machine[int]]:
    n       = draw(st.integers(min_value=min_size, max_value=max_size))
    return [
        draw(machine_st())
        for _ in range(n)
    ]


@st.composite
def job_st(
    draw,
    max_usage: int = 255,
    max_arrival: int = 10,
) -> Job[int]:
    arrival_time = draw(st.integers(0, max_arrival))
    status = JobStatus.NotCreated if arrival_time != 0 else JobStatus.Pending
    return Job(
        length=1,
        usage=draw(st.integers(min_value=0, max_value=max_usage)),
        meta=Metadata(
            arrival_time=arrival_time
        ),
        status=status,
    )


@st.composite
def jobs_st(
        draw,
        min_size: int = 1,
        max_size: int = 10,
) -> list[Job[int]]:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(job_st()) for _ in range(n)]
