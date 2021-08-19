# # This is homework.
# # Load your model and submit this to Jidi

import torch
import os
import numpy as np

# load critic
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic
import torch

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, id, obs_dim):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    observations = np.zeros((1, obs_dim)) # todo
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    agents_index = [id]
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions # todo: to check
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, element, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations.squeeze().tolist()

# TODO
state_dim = 18
hidden_size = 64
action_dim = 4
n_player = 2

class Ensemble(torch.nn.Module):
    def __init__(self, critic_eval):
        super().__init__()
        self.critic_eval = critic_eval
    def forward(self, x):
        return self.critic_eval[0](x)+self.critic_eval[1](x)


class IQL:
    def __init__(self):
        self.critic_eval = [Critic(state_dim, action_dim, hidden_size) for i in range(2)]
        self.ensemble_critic = Ensemble(self.critic_eval)


    def choose_action(self,observations):
        action = torch.argmax(self.ensemble_critic(torch.tensor(get_observations(observations,0,state_dim), dtype=torch.float).view(1,-1))).item()
        return action

    def load(self, path):
        for i in range(n_player):
            self.critic_eval[i].load_state_dict(torch.load(path+'/critic'+str(i)+'.pth'))


#TODO
def action_from_algo_to_env(joint_action):
    joint_action_ = []
    for a in range(1):
        action_a = joint_action
        each = [0] * action_dim
        each[action_a] = 1
        joint_action_.append(each)
    return joint_action_

# todo
# Once start to train, u can get saved model. Here we just say it is critic.pth.
critic_net = os.path.dirname(os.path.abspath(__file__))
agent = IQL()
agent.load(critic_net)


# todo
def my_controller(observation, action_space, is_act_continuous=False):
    obs = observation
    action = agent.choose_action(obs)
    return action_from_algo_to_env(action)