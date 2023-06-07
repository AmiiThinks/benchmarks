import gymnasium as gym
import popgym  # noqa: F401
import numpy as np

from gymnasium.wrappers.flatten_observation import FlattenObservation

class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    """
    A wrapper that converts a multidiscrete action space to discrete
    by taking the Cartesian product of all possible actions.
    """
    def __init__(self, env):
        super().__init__(env)
        # Get the number of actions for each dimension
        self.nvec = env.action_space.nvec
        # Compute the total number of discrete actions
        self.n = np.prod(self.nvec)
        # Create a discrete action space
        self.action_space = gym.spaces.Discrete(self.n)

    def action(self, action):
        # Convert the discrete action to a multidimensional index
        index = np.unravel_index(action, self.nvec)
        # Return the original action
        return index


def create_popgym_env(**env_config):
    env=gym.make(env_config['name'])
    if env.observation_space.__class__.__name__=='MultiDiscrete' or env.observation_space.__class__.__name__=='Discrete':
        env = FlattenObservation(env)
        #print("Using FlattenObservation because of MultiDiscrete observation space")
    if env.action_space.__class__.__name__=='MultiDiscrete':
        env = MultiDiscreteToDiscreteWrapper(env)
    return env