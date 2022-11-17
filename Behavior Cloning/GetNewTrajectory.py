import time
import copy
import math
import numpy as np
from GridEnv import GridEnv
import numpy as np
from tensorflow import keras
import pygame

def get_training_data(block_type, block_types_grid, block_ids_grid, shape_type, num_samples=1, test_data=[]):
    samples = []
    prev_traj = test_data[0] if len(test_data) > 0 else []
    for m in range(num_samples):
        samples.append(get_expert_trajectory(block_type, block_types_grid, block_ids_grid, shape_type, prev_traj))
    return samples

def get_expert_trajectory(block_type, block_types_grid, block_ids_grid, shape_type, test_data=[]):
    trajectory = []
    state = copy.deepcopy([block_types_grid, block_ids_grid, [block_type, shape_type]])
    trajectory.append(state)
    if len(test_data) > 0:
        trajectory = test_data + trajectory
    return trajectory


def prepare_data(data):
    blocks = []
    for seq in data:
        for idx in range(len(seq)):
                block_type = np.array(seq[idx][0])
                block_id = np.array(seq[idx][1])
                info = np.array(seq[idx][2])
                blocks_temp = np.append(block_type.flatten(), block_id.flatten())
                blocks_temp = np.append(blocks_temp, info.flatten())
                blocks.append(np.array(blocks_temp))
    return np.array(blocks)

model = keras.models.load_model('final_model')


# train = list(np.load('train_data.npy', allow_pickle=True))
# data  = prepare_data([train[0]])
# pred = model.predict(data)
# for i in pred:
#     l = [round(n) for n in i]
#     print(l)
        
# exit()
env = GridEnv()
block_type = 2
shape_type = 0  # 0 for tower and 1 for square
block_num_dict = {
    1: env.red_num,
    2: env.yellow_num,
    3: env.blue_num,
    4: env.pink_num
}
block_num = block_num_dict[block_type]
done = False


block_ids = [i for i in range(1,block_num+1)]
pred_id = 0
pred_type = 0
pred_x = 30
pred_y=0
test_data =[]
# time.sleep(5)
while not done:
    time.sleep(1)
    if len(block_ids) > 0 and pred_x > 0:
        test_data = get_training_data(block_type, env.block_types_grid, env.block_ids_grid, 0, 1, test_data)
        data = prepare_data(test_data)
        pred = model.predict(data)
        pred = pred[-1]
        pred_type = round(pred[0])
        pred_id = round(pred[1])
        pred_x = round(pred[2])
        pred_y=round(pred[3])
        print(pred_type, pred_id, pred_x,pred_y)
        env.step([pred_type, pred_id, pred_x, pred_y])
        block_ids.remove(pred_id)
    else:
        done = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    env.update()






