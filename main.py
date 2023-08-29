
from Snake import *
import gymnasium as gym
import numpy as np
import pygame

from stable_baselines3 import PPO



class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'array']}
    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.snake = Snake()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.MultiDiscrete(3 * np.ones((SIZE * SIZE)))

    def step(self, action):
        state, reward, terminated, truncated, info = self.snake.step(action)
        state = self.snake.get_observation().flatten()
        return state, reward, terminated, truncated, info

    def render(self, info):
        if self.render_mode == 'array':
            print(self.snake.get_observation().reshape((SIZE, SIZE)))
        if self.render_mode == 'human':
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
                pygame.display.set_caption('Snake')
                self.clock = pygame.time.Clock()
                self.score = 0
                self.font = pygame.font.SysFont('arial', 30)
            else:
                self.window.fill(WHITE)
                for l in self.snake.location:
                    pygame.draw.rect(self.window, GREEN, (l[0]*PIXEL, l[1]*PIXEL, PIXEL - 1, PIXEL - 1))
                pygame.draw.rect(self.window, RED, (self.snake.food[0]*PIXEL, self.snake.food[1]*PIXEL, PIXEL, PIXEL))
                if len(info) > 0:
                    self.score += 1
                text = self.font.render('Score: ' + str(self.score), True, BLACK)
                self.window.blit(text, [0, 0])
                pygame.display.flip()
                self.clock.tick(FPS)

    def reset(self, seed=None):
        self.snake = Snake()
        return self.snake.get_observation().flatten(), {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def env_test(env, model, episodes):
    for episode in range(1, episodes + 1):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        info = {}
        score = 0
        while not terminated:
            if model is not None:
                action, _ = model.predict(observation)
            else:
                action = ACTIONS[env.action_space.sample()]
            observation, reward, terminated, truncated, info = env.step(action)
            env.render(info)
            score += reward
        print("Episode: {}\n Score: {}".format(episode, score))
    env.close()