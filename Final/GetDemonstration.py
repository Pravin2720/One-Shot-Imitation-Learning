import copy
import math
import numpy as np
import time
from GridEnv import GridEnv
import pygame


def get_training_data(block_type, num_samples=100, shape_type=0):
    samples = []
    for m in range(num_samples):
        # time.sleep(1)
        samples.append(get_expert_trajectory(block_type, shape_type))

    return samples

def get_expert_trajectory(block_type, shape_type):
    trajectory = []
    env = GridEnv()
    done = False
    block_id = 1
    pos = [9,10]
    block_num_dict = {
        1: env.red_num,
        2: env.yellow_num,
        3: env.blue_num,
        4: env.pink_num
    }
    block_num = block_num_dict[block_type]

    if shape_type == 0:
        # For tower
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
    else:
        # For square
        side_length = math.floor(math.sqrt(block_num))
        total_required_num = side_length * side_length
        curr_num = 1
        direction = 1 # 1 for right 2 for top 3 for left
        while not done:
            # time.sleep(1)
            if curr_num <= total_required_num:
                print(pos)
                state = copy.deepcopy([env.block_types_grid, env.block_ids_grid, block_type, block_id, pos[0], pos[1],
                                    [block_type, shape_type]])
                trajectory.append(state)
                env.step([block_type, block_id, pos[0], pos[1]])
                curr_num += 1
                block_id += 1
                if direction == 1:
                    pos[1] += 1
                    if pos[1] - 10 == side_length:
                        direction = 3
                        pos[1] -= 1
                        pos[0] -= 1
                elif direction == 3:
                    pos[1] -= 1
                    if pos[1] == 9:
                        direction = 1
                        pos[1] += 1
                        pos[0] -= 1
            else:
                done = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            env.update()
        return trajectory


train_red_tower = get_training_data(1,10000, 0)
train_yellow_tower = get_training_data(2,10000, 0)
train_blue_tower = get_training_data(3,10000, 0)
train_pink_tower = get_training_data(4,10000, 0)



train_red_square = get_training_data(1,10000, 1)
train_yellow_square = get_training_data(2,10000, 1)
train_blue_square = get_training_data(3,10000, 1)
train_pink_square = get_training_data(4,10000, 1)
# train = train_red
train = train_red_tower + train_yellow_tower + train_blue_tower + train_pink_tower + train_red_square + train_yellow_square + train_blue_square + train_pink_square
# np.save('training_data.npy', np.array(train, dtype=object), True)
