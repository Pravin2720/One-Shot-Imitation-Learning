import copy

import numpy as np
import time
from GridEnv import GridEnv
import pygame
import random


def get_training_data(block_type, num_samples=100):
    samples = []
    for m in range(num_samples):
        # time.sleep(1)
        samples.append(get_expert_trajectory(block_type))
    return samples

def get_expert_trajectory(block_type):
    shape_type = 0 # 0 for tower and 1 for square
    trajectory = []
    env = GridEnv()

    block_num_dict = {
        1: env.red_num,
        2: env.yellow_num,
        3: env.blue_num,
        4: env.pink_num
    }
    block_num = block_num_dict[block_type]
    done = False
    pos = [9,11]
    # block_ids = [i for i in range(1,block_num+1)]
    # random.shuffle(block_ids)
    # block_id = block_ids.pop()
    block_id = 1


    while not done:
        # time.sleep(1)
        if block_id <= block_num and pos[0] >= 0:
            state = copy.deepcopy([env.block_types_grid, env.block_ids_grid, block_type, block_id, pos[0], pos[1], [block_type, shape_type]])
            trajectory.append(state)
            env.step([block_type,block_id,pos[0], pos[1]])
            # block_id = block_ids.pop()
            block_id += 1
            pos[0] -= 1
            # state = copy.deepcopy([env.block_types_grid, env.block_ids_grid, block_type, block_id, pos[0], pos[1], [block_type, shape_type]])
            # trajectory.append(state)
        else:
            done = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        env.update()
    return trajectory



train_red = get_training_data(1,10000)
train_yellow = get_training_data(2,10000)
train_blue = get_training_data(3,10000)
train_pink = get_training_data(4,10000)
# train_red = get_training_data(1,100)
# print(train_red)
# train = train_red
# train_yellow = get_training_data(2,100)
train = train_red + train_yellow + train_blue + train_pink
np.save('train_data.npy', np.array(train, dtype=object), True)

# test_blue = get_training_data(3, 100)
# np.save('test_blue.npy', np.array(test_blue, dtype=object), True)
