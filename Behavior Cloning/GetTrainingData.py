import numpy as np
import time
from GridEnv import GridEnv
import pygame

def get_training_data(num_samples=100):
    samples = []
    for m in range(num_samples):
        # time.sleep(1)
        samples.append(get_expert_trajectory())
    return samples


def get_expert_trajectory():
    trajectory = []
    env = GridEnv()
    rn = env.red_num
    done = False
    pos = [9,11]
    block_type = 1
    block_id = 1
    while not done:
        # time.sleep(1)
        if block_id <= rn and pos[0] >= 0:
            env.step([block_type,block_id,pos[0], pos[1]])
            trajectory.append([env.block_types_grid, env.block_ids_grid, block_type, block_id, pos[0], pos[1]])
            block_id += 1
            pos[0] -= 1
        else:
            done = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        env.update()
    return trajectory



train = get_training_data(100)
train = train
# print(len(train))
# np.save('new_data.npy', np.array(train, dtype=object), True)
