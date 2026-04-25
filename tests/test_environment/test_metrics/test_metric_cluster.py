import numpy as np
from hypothesis import given, strategies as st, assume, HealthCheck, settings

from environment.core.allocator import AllocationStatus
from environment.core.jobs import JobStatus
from environment.envs.metrics import Machines
from environment.envs.metrics.metric_cluster import Jobs, MetricResourceManagement
from tests.test_environment.test_metrics.strategies import cluster_st


@given(cluster=cluster_st())
def test_simple_cluster(cluster: MetricResourceManagement):
    assert cluster.jobs[0].usage.shape == cluster.machines[0].usage.shape
    n_resources, n_ticks = cluster.jobs[0].usage.shape
    assert cluster.current_tick == 0
    assert all(j.status in (JobStatus.NotCreated, JobStatus.Pending) for j in cluster.jobs)
    assert all( 0 < j.length <= n_ticks  for j in cluster.jobs)
    assert all(j.meta.run_time == 0 for j in cluster.jobs)
    assert all(j.meta.arrival_time >= 0 for j in cluster.jobs)

@given(cluster=cluster_st(), data=st.data())
def test_allocation_success(cluster: MetricResourceManagement, data):
    m_idx = data.draw(st.integers(0, len(cluster.machines)-1))
    j_idx = data.draw(st.integers(0, len(cluster.jobs)-1))
    machine = cluster.machines[m_idx]
    job = cluster.jobs[j_idx]
    assume(job.status is JobStatus.Pending)
    free_space = machine.capacity - machine.usage
    assume(np.all(free_space >= job.usage))
    allocation_status = cluster.allocate(machine, job)
    assert allocation_status is AllocationStatus.SUCCESS

@given(cluster=cluster_st(), data=st.data())
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_allocation_insufficient_resources(cluster: MetricResourceManagement, data):
    m_idx = data.draw(st.integers(0, len(cluster.machines) - 1))
    j_idx = data.draw(st.integers(0, len(cluster.jobs) - 1))
    machine = cluster.machines[m_idx]
    job = cluster.jobs[j_idx]
    assume(job.status is JobStatus.Pending)
    free_space = machine.capacity - machine.usage
    assume(np.any(free_space < job.usage))
    allocation_status = cluster.allocate(machine, job)
    assert allocation_status is AllocationStatus.INSUFFICIENT_RESOURCES

@given(cluster=cluster_st(), data=st.data())
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_allocation_incorrect_job_status_resources(cluster: MetricResourceManagement, data):
    m_idx = data.draw(st.integers(0, len(cluster.machines) - 1))
    j_idx = data.draw(st.integers(0, len(cluster.jobs) - 1))
    machine = cluster.machines[m_idx]
    job = cluster.jobs[j_idx]
    assume(job.status is not JobStatus.Pending)
    allocation_status = cluster.allocate(machine, job)
    assert allocation_status is AllocationStatus.UN_ALLOCATABLE_JOB
