# # This is homework.
# # Load your model and submit this to Jidi


import numpy as np
import os
import argparse
from common.utils import *
import time


# -------------------------------submission.py----------------------------#

# # This is homework.
# # Load your model and submit this to Jidi

import torch
import os

# load critic
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic


# TODO: Complete DQN algo under evaluation.
class DQN:
    def __init__(self):
        self.critic_eval = Critic(state_dim, action_dim, hidden_size)

    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))


def action_from_algo_to_env(joint_action):
    joint_action_ = []
    for a in range(n_player):
        action_a = joint_action
        each = [0] * action_dim
        each[action_a] = 1
        joint_action_.append(each)
    return joint_action_


n_player = 1
state_dim = 4
action_dim = 2
hidden_size = 64

# TODO: Once start to train, u can get saved model. Here we just say it is critic.pth.
critic_net = os.path.dirname(os.path.abspath(__file__)) + '/critic.pth'
agent = DQN()
agent.load(critic_net)


# This function dont need to change.
def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation['obs']
    action = agent.choose_action(obs)
    return action_from_algo_to_env(action)

# -------------------------------submission.py----------------------------#


if __name__ == '__main__':
    # set env and algo
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="gridworld", type=str,
                        help="gridworld/cliffwalking")
    parser.add_argument('--algo', default="tabularq", type=str,
                        help="tabularq/sarsa")

    parser.add_argument('--reload_config', action='store_true')
    args = parser.parse_args()

    print("================== args: ", args)
    print("== args.reload_config: ", args.reload_config)
    env = make_env(args)
    for i in range(100):
        obs, done = env.reset(), False
        tot_r = 0
        while not done:
            if env.env.n_player == 1:
                action = [my_controller(obs[0], env.env.joint_action_space[0])]  # single-player
            else:
                raise NotImplementedError  # multi-player
            obs, reward, done, _, info = env.step(action)
            env.make_render()
            time.sleep(0.1)
            tot_r += reward
        print('total_reward:', tot_r)