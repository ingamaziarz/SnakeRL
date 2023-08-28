from SnakeGame import SnakeGame
import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

SIZE = 10
ACTIONS = [-1, 0, 1]
class SnakeEnv(gym.Env): #SnakeEnv dziedziczy z klasy Env (jest podklasą Env)
    def __init__(self):
        self.game = SnakeGame()
        self.action_space = Discrete(3)
        self.observation_space = MultiDiscrete(3 * np.ones((SIZE, SIZE)))
        self.episode_length = 100
    def step(self, action):
        return self.game.step(action)
    def render(self):
        return self.game.get_observation()
    def reset(self):
        return self.game.get_observation()



episodes = 5
env = SnakeEnv()
# action and observation space
for episode in range(1, episodes + 1):
    obs = env.reset()
    terminated = False
    truncated = False
    info = {}
    score = 0

    while not terminated:
        action = ACTIONS[env.action_space.sample()]
        print("test")
        print(env.step(action))
        print("teset")
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        score += reward
        print(score)
    print("Episode: {}\n Score: {}".format(episode, score))
env.close()