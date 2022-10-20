import numpy as np
import os
import gym
import copy
import sys
import pygame
from gym.spaces import Discrete
import time


BLACK = (0, 0, 0)
YELLOW = (250, 250, 0)
RED= (255,0, 0)
BLUE= (0,0,255)
GRAY= (128,128,128)
PINK= (255,105,180)
colors=[RED, BLUE, PINK]
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400


class GridEnv(gym.Env):
    def __init__(self):
        # blocks order
        self.final_block_states = [[4,0],[4,1],[4,2]]
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

        # agent states
        self.start_state = self.get_agent_states()
        self.agent_state = copy.deepcopy(self.start_state)

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

    def get_agent_states(self):
        start_state = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 4)))
        if start_state == [None, None]:
            sys.exit('Start or Target state not specified')
        return start_state

    def get_block_states(self):
        block_state1 = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 1)))
        block_state2 = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 2)))
        block_state3 = list(map(lambda x: x[0] if len(x) > 0 else None, np.where(self.current_map == 3)))

        if block_state1 == [None, None] or block_state2 == [None, None] or block_state3 == [None, None]:
            sys.exit('Blocks position not specified')
        return [block_state1, block_state2, block_state3]

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
        for x in range(0, WINDOW_HEIGHT, block_size):
                for y in range(0, WINDOW_WIDTH, block_size):
                    if self.agent_state[0] * 100 == x and self.agent_state[1] * 100 == y:
                        rect = pygame.Rect(self.agent_state[0] * 100, self.agent_state[1] * 100, block_size,
                                           block_size)
                        pygame.draw.rect(self.SCREEN, YELLOW, rect, 1)
                    else:
                        rect = pygame.Rect(x, y, block_size, block_size)
                        pygame.draw.rect(self.SCREEN, BLACK, rect, 1)
        blocks = self.get_block_states()
        for i in range(len(blocks)) :
            x = blocks[i][0]*100
            y = blocks[i][1]*100
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
        curr_state = np.sum([self.agent_state, self.actions_pos_dict[self.actions[action]]], axis=0)

        if curr_state[0] < 0 or curr_state[0] > self.grid_shape[0] - 1 or curr_state[1] < 0 or curr_state[1] > self.grid_shape[1] - 1:
            return self.agent_state, 0, False, {}

        self.agent_state = curr_state

        done = True
        for index, block in enumerate(self.block_states):
            if block[0] != self.final_block_states[index][0] or block[1] != self.final_block_states[index][1]:
                done = False
                break
        info = {}
        return self.agent_state, 0, done, info

    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        self.current_map = copy.deepcopy(self.initial_map)

env=GridEnv()
env.render()
episodes = 1


for episode in range(1, episodes + 1):
    state = env.agent_state
    done = False
    print(state)
    for i in range(10):
        time.sleep(1)
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        env.render()
        print(env.actions[action])
        print(n_state)

