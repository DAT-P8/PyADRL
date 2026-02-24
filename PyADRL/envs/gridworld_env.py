"""Gridworld MA area defense environment"""

from pettingzoo import ParallelEnv

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "gridworld_environment_v0",
    }

    def __init__(self):
        self.id: int = None
        self.target_x: int = None
        self.target_y: int = None
        self.pursuer_x: list[int] = []
        self.pursuer_y: list[int] = []
        self.evader_x: list[int] = []
        self.evader_y: list[int] = []

    def reset(self, seed=None, options=None):

        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
