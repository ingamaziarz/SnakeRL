from SnakeEnv import *
from model import *
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os


def show_PPO(env):
    PPO_path = os.path.join('training', 'trained_models', 'PPO_snake')
    model = PPO.load(PPO_path, env=env)
    evaluation = evaluate_policy(model, env, n_eval_episodes=10)
    print("Average reward: {}\n Standard deviation: {}".format(evaluation[0], evaluation[1]))

def show_DQN(env):
    DQN_path = os.path.join('training', 'trained_models', 'DQN_snake')
    model = DQN.load(DQN_path, env=env)
    evaluation = evaluate_policy(model, env, n_eval_episodes=10)
    print("Average reward: {}\n Standard deviation: {}".format(evaluation[0], evaluation[1]))



def show_agent(env):
    agent_path = 'model/model.pt'
    model = torch.load(agent_path)
    rewards_list = env_test(env, 10, model, custom=True)
    print("Average reward: {}\n Standard deviation: {}".format(np.mean(rewards_list), np.std(rewards_list)))


if __name__=='__main__':
    env = SnakeEnv()
    show_DQN(env)
    show_PPO(env)
    show_agent(env)
