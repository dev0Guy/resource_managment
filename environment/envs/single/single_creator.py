import random

from environment.core.cluster import ClusterCreator
from environment.core.jobs import Metadata, JobStatus, Job
from environment.core.machines import Machine
from environment.envs.single.single_cluster import SingleResourceManagement, Machines, Jobs


class SingleResourceManagementRandomCreator(ClusterCreator):

    def __init__(self, n_machines: int, n_jobs: int):
        self.n_machines = n_machines
        self.n_jobs = n_jobs

    def create(self) -> SingleResourceManagement[Machines, Jobs]:
        return SingleResourceManagement(
            self._create_machines(),
            self._create_jobs()
        )

    def _create_machines(self) -> Machines:
        return [
            Machine(
                capacity=255,
                usage=0
            )
            for _ in range(self.n_machines)
        ]

    def _create_jobs(self) -> Jobs:
        return [
            Job(
                length=1,
                usage=random.randint(0, 255),
                meta=Metadata(
                    arrival_time=0
                ),
                status=JobStatus.Pending
            )
            for _ in range(self.n_jobs)
        ]

