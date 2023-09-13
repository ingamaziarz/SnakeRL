import pygame
import os
import random
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import torch.utils.tensorboard
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
#screen parameters
SIZE = 20
SPEED = 100
PIXEL = 20
W = PIXEL * SIZE

#colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

#direction
UP = np.array([0, -1]) * PIXEL
RIGHT = np.array([1, 0]) * PIXEL
DOWN = np.array([0, 1]) * PIXEL
LEFT = np.array([-1, 0]) * PIXEL
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

class SnakeEnv(gym.Env):

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.MultiDiscrete([2] * 11)
        self.window = None
        self.render()
        self.reset()

    def reset(self, seed=None):
        self.direction = RIGHT
        self.dir_idx = 1
        self.head = np.array([SIZE // 2, SIZE // 2]) * PIXEL
        self.snake = [self.head, self.head + LEFT, self.head + 2 * LEFT]
        self.score = 0
        self.place_food()
        self.frame_iteration = 0
        return self.get_state(), {}

    def step(self, action):
        self.frame_iteration += 1
        self.dir_idx = (self.dir_idx + action - 1) % 4
        self.direction = DIRECTIONS[self.dir_idx]
        self.snake.insert(0, self.head + self.direction)
        self.head = self.snake[0]

        reward = 0
        done = False

        if self.is_collision() or self.frame_iteration > SIZE * len(self.snake) * 2:
            done = True
            reward = -10
            return self.get_state(), reward, done, done, {'score': self.score}

        if np.array_equal(self.food, self.head):
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        self.render()
        self.clock.tick(SPEED)
        return self.get_state(), reward, done, done, {'score': self.score}

    def get_state(self):
        view = [self.head + DIRECTIONS[self.dir_idx], self.head + DIRECTIONS[(self.dir_idx + 1) % 4], self.head + DIRECTIONS[(self.dir_idx - 1) % 4]]
        dir = [np.array_equal(self.direction, direction) for direction in DIRECTIONS]
        return np.concatenate(([self.is_collision(point) for point in view],
                               dir, [self.food[0] < self.head[0], self.food[0] > self.head[0], self.food[1] < self.head[1], self.food[1] > self.head[1]])) * 1

    def place_food(self):
        self.food = np.array([random.randint(0, SIZE - 1) * PIXEL, random.randint(0, SIZE - 1) * PIXEL])
        if np.any(np.all(self.food == self.snake, axis=1)):
            self.place_food()
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if (pt[0] < 0 or pt[1] < 0 or pt[0] >= W or pt[1] >= W) or np.any(np.all(pt == self.snake[1:], axis=1)):
            return True
        return False

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((W, W))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        else:
            self.window.fill(WHITE)
            for pt in self.snake[1:]:
                pygame.draw.rect(self.window, GREEN, (pt[0], pt[1], PIXEL - 1, PIXEL - 1))
            pygame.draw.rect(self.window, DARK_GREEN, (self.head[0], self.head[1], PIXEL - 1, PIXEL - 1))
            pygame.draw.rect(self.window, RED, (self.food[0], self.food[1], PIXEL, PIXEL))

            text = pygame.font.SysFont('arial', 20).render("Score: " + str(self.score), True, BLACK)
            self.window.blit(text, [0, 0])
            pygame.display.flip()

    def close(self):
        pygame.quit()
        quit()

def env_test(env, episodes, model=None, custom=False):
    reward_per_episode_list = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        reward_per_episode = 0
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                if custom:
                    prediction = model(torch.tensor(obs, dtype=torch.float))
                    action = torch.argmax(prediction).item()
                else:
                    action, _ = model.predict(obs)
            obs, reward, done, _, score = env.step(action)
            reward_per_episode += reward
        print("Episode: {}\n Reward per episode: {}\n Score: {}".format(episode, reward_per_episode, score['score']))
        reward_per_episode_list.append(reward_per_episode)
    return reward_per_episode_list

if __name__=='__main__':
    env = SnakeEnv()
    check_env(env)
    episodes = 50

    log_path = os.path.join('training', 'logs')
    DQN_path = os.path.join('training', 'trained_models', 'DQN_snake')
    PPO_path = os.path.join('training', 'trained_models', 'PPO_snake')

    # model = PPO('MlpPolicy', env, verbose=1)
    # model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

    # model.learn(total_timesteps=int(1e5), reset_num_timesteps=False)

    # model.save(DQN_path)
    # model.save(PPO_path)





