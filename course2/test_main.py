# # This is homework.
# # Load your model and submit this to Jidi


import numpy as np
import os
import argparse
from common.utils import *

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



if __name__ == '__main__':
    # set env and algo
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="gridworld", type=str,
                        help="gridworld/cliffwalking")
    parser.add_argument('--algo', default="tabularq", type=str,
                        help="tabularq/sarsa")

    parser.add_argument('--reload_config', action='store_true')  # 加是true；不加为false
    args = parser.parse_args()

    print("================== args: ", args)
    print("== args.reload_config: ", args.reload_config)
    env = make_env(args)
    print('h',env.get_actionspace())
    for i in range(100):
        obs,done = env.reset(),False
        while not done:
            action = [my_controller(obs[0],1)]
            import time
            time.sleep(0.1)
            obs,reward,done,_,info = env.step(action)
            env.make_render()