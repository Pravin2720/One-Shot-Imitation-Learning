import numpy as np
import os
import gym
import copy
import sys
import pygame

BLACK = (0, 0, 0)
YELLOW = (250, 250, 0)
WINDOW_HEIGHT = 300
WINDOW_WIDTH = 300


class GridEnv(gym.Env):
    def __init__(self):
        # action space
        self.actions = ['up', 'down', 'right', 'left', 'begin']
        self.actions_pos_dict = {'up': [-1, 0], 'down': [1, 0], 'right': [0, -1], 'left': [0, 1], 'begin': [0, 0]}

        # construct the grid
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.insert_grid_map = os.path.join(file_path, 'map3.txt')
        self.initial_map = self.read_grid_map(self.insert_grid_map)
        self.current_map = copy.deepcopy(self.initial_map)
        self.grid_shape = self.initial_map.shape

        # agent actions
        self.start_state = self.get_agent_states()
        self.agent_state = copy.deepcopy(self.start_state)

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
                        pygame.draw.rect(SCREEN, YELLOW, rect,1)


        while True:
            drawGrid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()



    def step(self, action):
        print("step")

    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        self.current_map = copy.deepcopy(self.initial_map)

grid=GridEnv()
grid.render()