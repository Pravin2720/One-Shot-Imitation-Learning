import numpy as np
import os
import gym
import copy
import sys
import pygame
from gym.spaces import Discrete


BLACK = (0, 0, 0)
YELLOW = (250, 250, 0)
WINDOW_HEIGHT = 300
WINDOW_WIDTH = 300


class GridEnv(gym.Env):
    def __init__(self):
        # blocks order
        self.final_block_states = [[4,0],[4,1],[4,2]]
        # action space
        self.actions = ['up', 'down', 'right', 'left']
        self.actions_pos_dict = {'up': [0, -1], 'down': [0, 1], 'right': [1, 0], 'left': [-1, 0], 'begin': [0, 0]}
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
        SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        SCREEN.fill(BLACK)

        def drawGrid():
            height = self.grid_shape[0]
            width = self.grid_shape[1]
            block_size = WINDOW_HEIGHT//height
            # blockSize = 100  # Set the size of the grid block
            for x in range(0, WINDOW_HEIGHT, block_size):
                    for y in range(0, WINDOW_WIDTH, block_size):
                        rect = pygame.Rect(x , y , block_size, block_size)
                        pygame.draw.rect(SCREEN, YELLOW, rect, 1)


        while True:
            drawGrid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()



    def step(self, action):
        curr_state = np.sum([self.agent_state, self.actions_pos_dict[self.actions[action]]], axis=0)

        if curr_state[0] < 0 or curr_state[0] > self.grid_shape[0] or curr_state[1] < 0 or curr_state[1] > self.grid_shape[1]:
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
episodes = 1


for episode in range(1, episodes + 1):
    state = env.agent_state
    done = False
    for i in range(10):
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)

