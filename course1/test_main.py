# # This is homework.
# # Load your model and submit this to Jidi


import numpy as np
import os
import argparse
from common.utils import *
import time


# -------------------------------submission.py----------------------------#

# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agents is random agents , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""


def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    for i in range(len(action_space)):
        action_ = sample_single_dim(action_space[i], is_act_continuous)
        agent_action.append(action_)
    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each


def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = action_space_list_each[j].high
                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player

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