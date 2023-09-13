import os
import torch
import random
from collections import deque
from SnakeEnv import SnakeEnv
from model import Linear_QNet, QTrainer
from torch.utils.tensorboard import SummaryWriter



MAX_MEMORY = int(1e5)
BATCH_SIZE = 100
LR = 0.001

class Agent:

    def __init__(self, input_layer, output_layer):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(input_layer, 256, output_layer)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def train_memory(self):
        sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    def get_action(self, state):
        self.epsilon = 100 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            action = torch.argmax(prediction).item()
        return action

def train():
    log_path = os.path.join('training', 'logs', 'agent')
    writer = SummaryWriter(log_path)
    reward_sum = 0
    record = 0
    game = SnakeEnv()
    agent = Agent(game.observation_space.shape[0], game.action_space.n)
    iter = 0
    while iter < int(1e5):
        iter += 1
        state_old = game.get_state()
        final_move = agent.get_action(state_old)
        state_new, reward, done, _, score = game.step(final_move)
        agent.memory.append((state_old, final_move, reward, state_new, done))
        agent.trainer.train_step(state_old, final_move, reward, state_new, done)
        reward_sum += reward

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_memory()

            if score['score'] > record:
                record = score['score']
                agent.model.save()

            print('Game', agent.n_games, 'Score', score['score'], 'Record:', record)

            reward_average = reward_sum / agent.n_games
            writer.add_scalar('ep_rew_mean', reward_average, iter)

if __name__ == '__main__':
    # train()
    pass