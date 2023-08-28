import gym
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from Snake import Snake, SnakeException, DIRECTIONS, SIZE
import numpy as np

class SnakeGame:
    def __init__(self):
        self.snake = Snake()
        self.done = False

    def step(self, action):
        length = self.snake.length
        #actions: -1 left, 0 straight, 1 right
        self.snake.direction = DIRECTIONS[(DIRECTIONS.index(self.snake.direction) + action) % 4]
        try:
            self.snake.move()
        except SnakeException:
            print("exco")
            self.done = True
        if self.done:
            state = self.get_observation()
            return state, length, self.done, self.done, {}
        self.snake.eaten()
        state = self.get_observation()
        return state, length, self.done, False, {}

    def get_observation(self):
        screen = np.zeros((SIZE, SIZE))
        for s in self.snake.location: screen[s] = 1
        screen[self.snake.food] = 2
        return screen
    def reset(self):
        return self.get_observation()
