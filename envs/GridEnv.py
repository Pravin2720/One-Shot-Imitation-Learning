import numpy as np
import os
import gym
import copy
import sys
import pygame
from gym.spaces import Discrete
import time
import json



BLACK = (0, 0, 0)
YELLOW = (250, 250, 0)
RED= (255,0, 0)
BLUE= (0,0,255)
GRAY= (128,128,128)
PINK= (255,105,180)
colors=[RED, BLUE, PINK, YELLOW]
WINDOW_HEIGHT = 500
WINDOW_WIDTH = 500


class GridEnv(gym.Env):
    def __init__(self, block_nums):
        # Block nums
        self.block_nums = block_nums
        # blocks order
        self.final_block_states = [[4,0],[4,1],[4,2],[4,3]]
        # action space
        self.actions = ['up', 'down', 'right', 'left']
        self.actions_pos_dict = {'up': [-1, 0], 'down': [1, 0], 'right': [0, 1], 'left': [0, -1]}
        self.action_space = Discrete(4)
        # construct the grid
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.insert_grid_map = os.path.join(file_path, 'map3.txt')
        self.initial_map = self.read_grid_map(self.insert_grid_map)
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






env=GridEnv(4)
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

done = False
data = {
    "final_configurations": final_configurations.tolist(),
    "data" : []
}
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
                    data["data"].append({
                        "current_block": env.current_block,
                        "current_block_moves" : [],
                        "grid_states":[]
                    })
                    break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                data["data"][-1]["current_block_moves"].append(3)
                data["data"][-1]["grid_states"].append(env.current_map.tolist())
                n_state, reward, done, info = env.step(3)
            if event.key == pygame.K_RIGHT:
                data["data"][-1]["current_block_moves"].append(2)
                data["data"][-1]["grid_states"].append(env.current_map.tolist())
                n_state, reward, done, info = env.step(2)
            if event.key == pygame.K_UP:
                data["data"][-1]["current_block_moves"].append(0)
                data["data"][-1]["grid_states"].append(env.current_map.tolist())
                n_state, reward, done, info = env.step(0)
            if event.key == pygame.K_DOWN:
                data["data"][-1]["current_block_moves"].append(1)
                data["data"][-1]["grid_states"].append(env.current_map.tolist())
                n_state, reward, done, info = env.step(1)
            env.drawGrid()
            env.update()


print(data)
# Serializing json
json_object = json.dumps(data, indent=4)
# Writing to demonstration.json
with open("demonstration.json", "w") as outfile:
    outfile.write(json_object)

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






