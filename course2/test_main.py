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


import numpy as np
import os
import argparse

# todo
# Once start to train, u can get saved model. Here we just say it is q_table.pth.
q_table = os.path.dirname(os.path.abspath(__file__)) + '/q_table.pth'
q_values = np.loadtxt(q_table, delimiter=",")


def action_from_algo_to_env(joint_action):
    joint_action_ = []
    for a in range(1):
        action_a = joint_action
        each = [0] * 4
        each[action_a] = 1
        joint_action_.append(each)
    return joint_action_


# todo
def behaviour_policy(q):
    return np.argmax(q)


# todo
def epsilon_greedy(q_values):
    pass


# todo
def my_controller(observation, action_space, is_act_continuous=False):
    global q_values
    obs = observation['obs']
    action = behaviour_policy(q_values[obs])
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