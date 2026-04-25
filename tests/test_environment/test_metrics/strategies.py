import numpy as np
from hypothesis import strategies as st
import numpy.typing as npt

from environment.core.jobs import Job, Metadata, JobStatus
from environment.core.machines import Machine
import hypothesis.extra.numpy as hnp

from environment.envs.metrics import Machines
from environment.envs.metrics.metric_cluster import MetricResourceManagement


@st.composite
def machine_st(
    draw,
    n_resources: int,
    n_ticks: int,
    min_capacity: int
) -> Machine[npt.NDArray[np.uint8]]:
    capacity = draw(hnp.arrays(
        elements=st.integers(min_capacity, 255),
        shape=(n_resources, n_ticks),
        dtype=np.uint8)
    )
    usage = np.zeros_like(capacity, dtype=np.uint8)
    return Machine(capacity, usage)


@st.composite
def machines_st(
    draw,
    n_machines: int,
    n_resources: int,
    n_ticks: int,
    min_capacity: int
) -> Machines:
    return [
        draw(machine_st(n_resources, n_ticks, min_capacity))
        for _ in range(n_machines)
    ]

@st.composite
def job_st(
    draw,
    n_resources: int,
    n_ticks: int,
    max_usage: int
) -> Job[npt.NDArray[np.uint8]]:
    length = draw(st.integers(1, n_ticks))
    arrival_tick = draw(st.integers(0, max(1, n_ticks - length)))
    usage = draw(
        hnp.arrays(
            elements=st.integers(0, max_usage),
            shape=(n_resources, n_ticks),
            dtype=np.uint8
        )
    )
    usage[:, length-1:] = 0
    status = JobStatus.Pending if arrival_tick == 0 else JobStatus.NotCreated
    return Job(
        length=length,
        usage=usage,
        meta=Metadata(arrival_time=arrival_tick),
        status=status
    )

@st.composite
def jobs_st(
    draw,
    n_jobs: int,
    n_resources: int,
    n_ticks: int,
    max_usage: int
) -> Machines:
    return [
        draw(job_st(n_resources=n_resources, n_ticks=n_ticks, max_usage=max_usage))
        for _ in range(n_jobs)
    ]

@st.composite
def cluster_st(
    draw,
    max_n_machines: int = 5,
    max_n_jobs: int = 10,
    max_n_resources: int = 5,
    max_n_ticks: int = 50
) -> MetricResourceManagement:
    n_machines = draw(st.integers(1, max_n_machines))
    n_jobs = draw(st.integers(1, max_n_jobs))
    n_resources = draw(st.integers(1, max_n_resources))
    n_ticks = draw(st.integers(1, max_n_ticks))
    min_capacity = draw(st.integers(125, 255))
    max_usage = draw(st.integers(10, min_capacity))
    machines = draw(machines_st(n_machines, n_resources, n_ticks, min_capacity))
    jobs = draw(jobs_st(n_jobs, n_resources, n_ticks, max_usage))
    return MetricResourceManagement(machines, jobs)