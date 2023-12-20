
from .text_env import Env
try:
    import gym
    gym.register(
        id='BlockmazeTextEnv-v1',
        entry_point='text_crafter.text_blockmaze:Env',
        max_episode_steps=10000,
        kwargs={'reward': False, 'env_reward': False, 'seed': 1})
except ImportError:
    pass
