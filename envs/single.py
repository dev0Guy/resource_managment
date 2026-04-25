import gymnasium as gym


class SingleResourceManagementEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 0, 0, True, False, {}

    def render(self):
        pass

    def close(self):
        pass
