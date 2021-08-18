# # This is homework.
# # Load your model and submit this to Jidi


import numpy as np
import os
import argparse
from common.utils import *
import time


# -------------------------------submission.py----------------------------#

"""
    copy your submission.py code here and test your algorithm locally
"""

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