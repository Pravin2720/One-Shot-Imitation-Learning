import numpy as np
import os
import gym
import copy
import sys
import pygame
from gym.spaces import Discrete
import time
import json
import math

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


BLACK = (0, 0, 0)
YELLOW = (250, 250, 0)
RED= (255,0, 0)
BLUE= (0,0,255)
GRAY= (128,128,128)
PINK= (255,105,180)
colors=[RED, BLUE, PINK, YELLOW]
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400


class GridEnv(gym.Env):
    def __init__(self, block_nums):
        # Block nums
        self.block_nums = block_nums
        # blocks order
        # self.final_block_states = [[3,0],[3,1],[3,2],[3,3]]
        # self.final_block_states = [[3, 0]]
        self.final_block_states = [[np.random.randint(0, high=4), np.random.randint(0, high=4)]]
        # action space
        self.actions = ['up', 'down', 'right', 'left', 'begin']
        self.actions_pos_dict = {'up': [-1, 0], 'down': [1, 0], 'right': [0, 1], 'left': [0, -1], 'begin': [0, 0]}
        self.action_space = Discrete(5)
        # construct the grid
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.insert_grid_map = os.path.join(file_path, 'map3.txt')
        self.initial_map = self.read_grid_map(self.insert_grid_map)
        x, y = np.random.randint(0, high=4), np.random.randint(0, high=4)
        while x == self.final_block_states[0][0] and y == self.final_block_states[0][1]:
            x, y = np.random.randint(0, high=4), np.random.randint(0, high=4)
        self.initial_map[x,y] = 1
        self.current_map = copy.deepcopy(self.initial_map)
        self.grid_shape = self.initial_map.shape

        self.current_block = None
        self.current_block_pos = None
        # agent states
        # self.start_state = self.get_agent_states()
        # self.agent_state = copy.deepcopy(self.start_state)

        # block states
        self.initial_block_states = self.get_block_states()
        self.block_states = copy.deepcopy(self.initial_block_states)


        # env parameters
        self.reset()
        self.render()


    def read_grid_map(self, insert_grid_map):
        with open(insert_grid_map, 'r') as f:
            grid_map = f.readlines()
            grids = np.array(list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), grid_map)))
            return grids

    # def get_agent_states(self):
    #     start_state = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 9)))
    #     if start_state == [None, None]:
    #         sys.exit('Start or Target state not specified')
    #     return start_state

    def get_block_states(self):
        block_states = []
        for i in range(1,self.block_nums+1):
            block_state = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == i)))
            if block_state == [None, None]:
                sys.exit('Blocks position not specified')
            block_states.append(block_state)
        return block_states
        # block_state1 = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 1)))
        # block_state2 = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 2)))
        # block_state3 = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 3)))
        # if block_state1 == [None, None] or block_state2 == [None, None] or block_state3 == [None, None]:
        #     sys.exit('Blocks position not specified')
        # return [block_state1, block_state2, block_state3]

    def get_block_state(self, block):
        block_state = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == block)))
        if block_state == [None, None]:
            sys.exit('Block not found!')
        return block_state

    def render(self):
        self.SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.SCREEN.fill(BLACK)
        self.drawGrid()
        self.update()

    def drawGrid(self):
        height = self.grid_shape[0]
        width = self.grid_shape[1]
        block_size = WINDOW_HEIGHT//height
        # blockSize = 100  # Set the size of the grid block
        for x in range(0, WINDOW_WIDTH, block_size):
                for y in range(0, WINDOW_HEIGHT, block_size):
                    # if self.agent_state[0] * 100 == x and self.agent_state[1] * 100 == y:
                    #     rect = pygame.Rect(self.agent_state[0] * 100, self.agent_state[1] * 100, block_size,
                    #                        block_size)
                    #     pygame.draw.rect(self.SCREEN, YELLOW, rect, 1)
                    # else:
                    rect = pygame.Rect(x, y, block_size, block_size)
                    pygame.draw.rect(self.SCREEN, BLACK, rect)
        blocks = self.get_block_states()
        for i in range(len(blocks)):
            x = blocks[i][1]*100
            y = blocks[i][0]*100
            # print(x,y)

            rect = pygame.Rect(x, y, block_size, block_size)
            pygame.draw.rect(self.SCREEN, colors[i], rect)
        for block in self.final_block_states:
            x = block[1]*100
            y = block[0]*100
            # print(x,y)

            rect = pygame.Rect(x, y, block_size, block_size)
            pygame.draw.rect(self.SCREEN, BLUE, rect)


    def update(self):
        self.drawGrid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        # while True:
        #     drawGrid()



    def step(self, action):
        curr_state = np.sum([self.current_block_pos, self.actions_pos_dict[self.actions[action]]], axis=0)
        if curr_state[0] < 0 or curr_state[0] > self.grid_shape[0] - 1 or curr_state[1] < 0 or curr_state[1] > self.grid_shape[1] - 1:
            return self.current_block_pos, 0, False, {}

        current_blocks_pos = self.get_block_states()

        for i in range(1, self.block_nums + 1):
            if i == self.current_block:
                continue
            block = current_blocks_pos[i - 1]
            if block[0] == curr_state[0] and block[1] == curr_state[1]:
                return self.current_block_pos, 0, False, {}

        # prev_block = current_blocks_pos[((self.current_block - 2) % 3)]
        # next_block = current_blocks_pos[((self.current_block) % 3)]
        #
        # if prev_block[0] == curr_state[0] and prev_block[1] == curr_state[1]:
        #     return self.current_block_pos, 0, False, {}
        # if next_block[0] == curr_state[0] and next_block[1] == curr_state[1]:
        #     return self.current_block_pos, 0, False, {}

        self.current_map[self.current_block_pos[0], self.current_block_pos[1]] = 0
        self.current_block_pos = curr_state
        self.current_map[self.current_block_pos[0], self.current_block_pos[1]] = self.current_block
        self.block_states = self.get_block_states()
        done = True
        for index, block in enumerate(self.block_states):

            if block[0] != self.final_block_states[index][0] or block[1] != self.final_block_states[index][1]:
                done = False
                break
        info = {}
        return self.current_block_pos, 0, done, info

    def reset(self):
        # self.agent_state = copy.deepcopy(self.start_state)
        self.current_map = copy.deepcopy(self.initial_map)
        # self.block_states = copy.deepcopy(self.initial_block_states)






env=GridEnv(1)
env.render()
episodes = 1


def read_configurations( configurations):
    with open(configurations, 'r') as f:
        configuration = f.readlines()
        final_configurations = np.array(list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), configuration)))
        return final_configurations

# configurations
file_path = os.path.dirname(os.path.realpath(__file__))
configurations = os.path.join(file_path, 'configurations.txt')
final_configurations = read_configurations(configurations)

# data = {
#     "final_configurations": final_configurations.tolist(),
#     "data" : []
# }
def get_training_data(num_samples=1):
    trajectory_batches = []  # The collecton of trajectories
    num_samples = num_samples
    for m in range(num_samples):
        trajectory_batches.append(get_expert_trajectory())
    # plt.imshow(self.moving_frame)
    return trajectory_batches

def get_expert_trajectory():
    data = []
    done = False
    block_states = env.get_block_states()
    moving_frame = copy.deepcopy(env.current_map)
    data.append([moving_frame, block_states, env.final_block_states, 4])
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONUP:
                pos = list(pygame.mouse.get_pos())
                y = pos[0] // 100
                x = pos[1] // 100
                block_states = env.get_block_states()
                for index, block in enumerate(block_states):
                    if x == block[0] and y == block[1]:
                        env.current_block = index + 1
                        env.current_block_pos = env.get_block_state(env.current_block)
                        break
            elif event.type == pygame.KEYDOWN:
                step = None
                if event.key == pygame.K_LEFT:
                    step = 3
                if event.key == pygame.K_RIGHT:
                    step = 2
                if event.key == pygame.K_UP:
                    step = 0
                if event.key == pygame.K_DOWN:
                    step = 1
                if step in env.action_space:
                    n_state, reward, done, info = env.step(step)
                    block_states = env.get_block_states()
                    moving_frame = copy.deepcopy(env.current_map)
                    data.append([moving_frame, block_states, env.final_block_states, step])

                env.drawGrid()
                env.update()
    return data

max_seqlen = 30
def prepare_data(t, test):
    x_data = []
    y_data = []
    seq_len = []
    for p in t:
        eg_x = []
        eg_y = []
        seq_len.append(len(p))
        for idx in range(max_seqlen):
            if idx < len(p):
                xxx = []
                yyy = []
                for c in p[idx][1][0]:
                    xxx.append((c - 2) / 2)
                for c in p[idx][2][0]:
                    yyy.append((c - 2) / 2)
                # for cd in p[idx][1]:
                #     xx = []
                #     for c in cd:
                #         xx.append((c - 2) / 2)
                #     xxx.append(xx)
                # for cd in p[idx][2]:
                #     yy = []
                #     for c in cd:
                #         yy.append((c - 2) / 2)
                #     yyy.append(yy)
                eg_x.append(np.hstack([xxx, yyy]))
                eg_y.append(p[idx][3])
            else:
                eg_x.append(np.zeros((state_size)))
        x_data.append(eg_x)
        y_data.append(eg_y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    seq_len = np.array(seq_len)
    return x_data, y_data, seq_len


num_samp = 1
batch_size = 32
#

old_training_data = list(np.load('data_1.npy',allow_pickle=True))

train = get_training_data(num_samp)
train = train + old_training_data
print(len(train))
# for i in range(1999):
#     train.append(train[0])
np.save('data_1.npy', np.array(train, dtype=object), True)
exit()
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

tf.reset_default_graph()
num_hidden = 32
num_hidden1 = 24

state_size = 4
x = tf.placeholder('float32', [None, max_seqlen, state_size], name='X')
y = tf.placeholder('float32', [None, 1], name='Y')

W1 = tf.Variable(tf.random_normal([num_hidden + state_size, num_hidden1]))
b1 = tf.Variable(tf.random_normal([num_hidden1]))
W2 = tf.Variable(tf.random_normal([num_hidden1, 1]))
b2 = tf.Variable(tf.random_normal([1]))

seqlen = tf.placeholder('int32', [None], name='Seq_len')
size = tf.placeholder('int32', [1])
cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

# output, state = tf.nn.dynamic_rnn(cell, x, dtype='float32', sequence_length=seqlen)

def get_embedding(cell, data, seqlen):
    # tf.variable
    # with tf.variable_scope('model',reuse=None):
    output, state = tf.nn.dynamic_rnn(cell, data, dtype='float32', sequence_length=seqlen)
    b_size = tf.shape(output)[0]
    index = tf.range(0, b_size) * max_seqlen + (seqlen - 1)
    # Indexing
    last = tf.gather(tf.reshape(output, [-1, num_hidden]), index)
    return last, b_size


# last=val[5,:,:]
# Indexing
op, b_size = get_embedding(cell, x, seqlen)

def predict_action(x, embedding, seqlen):
    op_r = tf.tile(tf.reshape(embedding, (-1, num_hidden, 1)), [1, 1, max_seqlen])
    op_r = tf.transpose(op_r, [0, 2, 1])
    context = tf.concat((x, op_r), axis=2)
    a = tf.sequence_mask(seqlen, max_seqlen)
    context = tf.boolean_mask(context, a)
    context = tf.reshape(context, (-1, num_hidden + state_size))
    act1 = tf.nn.relu(tf.add(tf.matmul(context, W1), b1))
    action = tf.nn.tanh(tf.add(tf.matmul(act1, W2), b2))

    return action

action = predict_action(x, op, seqlen)
loss = tf.losses.mean_squared_error(y, action)
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train = list(np.load('data_1.npy',allow_pickle=True))
x_data, y_data, seq_len = prepare_data(train, 0)

x_d, y_d, y_r, seq_l = None, None, None, None
for m in range(10):
    epoch_loss = 0
    for k in range(int(num_samp / batch_size)):
        # print(k)
        ind = np.random.randint(0, x_data.shape[0], batch_size)
        x_d, y_d, seq_l = x_data[ind], y_data[ind], seq_len[ind]
        y_r = y_d.ravel().reshape(-1,1)
        _, cost_i = sess.run([optimizer, loss], feed_dict = { x: x_d,y: y_r, seqlen: seq_l})
        epoch_loss += cost_i
        # sess.run(optimizer, feed_dict={x: x_d, y: y_r, seqlen: seq_l})
    print('epoch number ' + str(m))
    # print(sess.run(loss, feed_dict={x: x_d, y: y_r, seqlen: seq_l}))
    print('Epoch', epoch_loss)

# After training, let us test on new samples
test = list(np.load('data_1.npy',allow_pickle=True))
x_test, y_test, seq_test = prepare_data([test[0]], 1)
y_test = y_test.ravel().reshape(-1, 1)
actions = (sess.run(action, feed_dict={x: x_test, y: y_test, seqlen: seq_test}))
print(actions)
# for i in range(10):
#     test = get_training_data(num_samples=1)
#     x_test, y_test, seq_test, image = prepare_data(test, 1)
#     y_test = y_test.ravel().reshape(-1, 1)
#     actions = (sess.run(action, feed_dict={x: x_test, y: y_test, seqlen: seq_test}))
#     print(actions)



# Serializing json
# json_object = json.dumps(data, indent=4)
# Writing to demonstration.json
# with open("demonstration.json", "w") as outfile:
#     outfile.write(json_object)

# for configuration in final_configurations:
#     for curr_block in configuration:
#         env.current_block = curr_block
#         env.current_block_pos = env.get_block_state(curr_block)
#         done = False
#         for i in range(10):
#             time.sleep(1)
#             action = env.action_space.sample()
#             n_state, reward, done, info = env.step(action)
#             if done:
#                 sys.exit("Success")
#             env.drawGrid()
#             env.update()
            # print(env.actions[action])
            # print(env.current_map)






