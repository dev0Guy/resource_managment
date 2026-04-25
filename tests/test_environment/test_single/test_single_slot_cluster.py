import random

from hypothesis import given

from environment.core.jobs import JobStatus
from environment.envs.single import Machines, Jobs
from environment.envs.single.single_cluster import SingleResourceManagement
from tests.test_environment.test_single.strategies import *


@given(machines=machines_st(), jobs=jobs_st())
def test_simple_cluster_creation(machines: Machines, jobs: Jobs):
    cluster = SingleResourceManagement(machines, jobs)
    assert cluster.machines == machines
    assert cluster.jobs == jobs
    assert cluster.current_tick == 0
    assert all(j.status in (JobStatus.NotCreated, JobStatus.Pending) for j in cluster.jobs)
    assert all(j.length == 1 for j in cluster.jobs)
    assert all(j.meta.run_time == 0 for j in cluster.jobs)
    assert all(j.meta.arrival_time >= 0 for j in cluster.jobs)

# TODO: test allocation
# TODO: test clock tick

@given(machines=machines_st(), jobs=jobs_st())
def test_simple_cluster_run_random_scheduler_until_completed(machines: Machines, jobs: Jobs):
    max_steps = 10_000
    rng = random.Random(0)
    cluster = SingleResourceManagement(machines, jobs)

    def is_finished():
        return all(j.status == JobStatus.Completed for j in cluster.jobs)

    for step in range(max_steps):

        if is_finished():
            break

        pending_jobs_idx = [
            idx for idx, j in enumerate(cluster.jobs)
            if j.status == JobStatus.Pending
        ]

        if not pending_jobs_idx:
            cluster.tick()
            continue

        job_idx = rng.choice(pending_jobs_idx)
        job = cluster.jobs[job_idx]

        possible_machines_idx = [
            idx for idx, m in enumerate(cluster.machines)
            if (m.capacity - m.usage) >= job.usage
        ]

        if not possible_machines_idx:
            cluster.tick()
            continue

        machine_idx = rng.choice(possible_machines_idx)
        machine = cluster.machines[machine_idx]

        result = cluster.allocate(machine, job)

        assert result is not None, "Allocation returned None unexpectedly"

    assert all(j.status == JobStatus.Completed for j in cluster.jobs)