from gymnasium import register, Env
from gymnasium.envs.registration import EnvCreator

from environment.envs.metrics import MetricResourceManagementEnvironment
from environment.envs.metrics.metric_creator import MetricResourceManagementHomogeneousCreator

class Generate(EnvCreator):

    def __call__(self, **kwargs) -> Env:
        creator = MetricResourceManagementHomogeneousCreator(
            n_machines=kwargs["n_machines"],
            n_jobs=kwargs["n_jobs"],
            n_resources=kwargs["n_resources"],
            n_ticks=kwargs["n_ticks"],
            offline=kwargs["offline"]
        )
        return MetricResourceManagementEnvironment(creator, render_mode=kwargs["render_mode"])


register(
    "resource-management-online",
    Generate(),
    kwargs=dict(
        n_machines=1,
        n_jobs=10,
        n_resources=3,
        n_ticks=5,
        offline=True,
        render_mode=None
    )
)
