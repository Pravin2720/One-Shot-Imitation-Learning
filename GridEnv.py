import numpy as np
import gym
import pygame
import sys
import random
import time

BLACK = (0, 0, 0)
YELLOW = (250, 250, 0)
RED= (255,0, 0)
BLUE= (0,0,255)
PINK= (255,105,180)
# colors=[BLACK, RED, BLUE, PINK, YELLOW]
colors=[BLACK, RED, YELLOW, BLUE, PINK]

WINDOW_HEIGHT = 1000
WINDOW_WIDTH = 1500

# move(blocktype, blockid, x,y)

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

class GridEnv(gym.Env):
    def __init__(self):
        self.total_blocks = 30
        # random no of blocks for each type
        self.red_num, self.yellow_num, self.blue_num, self.pink_num = constrained_sum_sample_pos(4,30)
        self.block_types = { "red" : 1, "yellow": 2, "blue": 3, "pink": 4}
        self.block_types_inv = { 1: "red", 2 : "yellow", 3: "blue", 4: "pink" }
        self.grid_shape = [10, 15]
        # Two grids
        # First to store the type(color) of block
        self.block_types_grid = np.zeros((10,15), dtype=int) #[[0] * self.grid_shape[1]] * self.grid_shape[0]
        # Second to store the id of the block to keep track of order
        self.block_ids_grid = np.zeros((10,15),dtype=int) # [[0] * self.grid_shape[1]] * self.grid_shape[0]
        self.generate_positions()

        self.print_block_ids()

        self.render()


    def render(self):
        pygame.init()
        self.SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.SCREEN.fill(BLACK)
        self.drawGrid()
        self.update()

    def drawGrid(self):
        height = self.grid_shape[0]
        width = self.grid_shape[1]
        block_size = WINDOW_HEIGHT//height
        for x in range(0, WINDOW_HEIGHT, block_size):
                for y in range(0, WINDOW_WIDTH, block_size):
                    rect = pygame.Rect(y, x, block_size, block_size)
                    x1 = (x // block_size)
                    y1 = (y // block_size)
                    color = colors[self.block_types_grid[x1][y1]]
                    pygame.draw.rect(self.SCREEN, color, rect)


    def update(self):
        self.drawGrid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

    def generate_positions(self):
        n = 0
        n_red = self.red_num
        n_yellow = self.yellow_num
        n_blue = self.blue_num
        n_pink = self.pink_num

        while n <= 30:
            x = random.randint(0, 9)
            y = random.randint(0, 9)
            block_type = 0
            block_id = 0
            if self.block_types_grid[x][y] == 0:
                if n_red > 0:
                    n_red -= 1
                    block_type = self.block_types["red"]
                    block_id = self.red_num - n_red
                    n += 1
                elif n_yellow > 0:
                    n_yellow -= 1
                    block_type = self.block_types["yellow"]
                    block_id = self.yellow_num - n_yellow
                    n += 1
                elif n_blue > 0:
                    n_blue -= 1
                    block_type = self.block_types["blue"]
                    block_id = self.blue_num - n_blue
                    n += 1
                elif n_pink > 0:
                    n_pink -= 1
                    block_type = self.block_types["pink"]
                    block_id = self.pink_num - n_pink
                    n += 1
                else:
                    n += 1
                self.block_types_grid[x][y] = block_type
                self.block_ids_grid[x][y] = block_id

    def print_block_ids(self):
        redn = []
        yellown = []
        pinkn = []
        bluen = []
        for x, r in enumerate(self.block_types_grid):
            for y, c in enumerate(r):
                if c == 1:
                    redn.append([self.block_ids_grid[x][y], x, y])
                if c == 2:
                    yellown.append([self.block_ids_grid[x][y], x, y])
                if c == 3:
                    bluen.append([self.block_ids_grid[x][y], x, y])
                if c == 4:
                    pinkn.append([self.block_ids_grid[x][y], x, y])
        print("red", sorted(redn))
        print("yellow", sorted(yellown))
        print("blue", sorted(bluen))
        print("pink", sorted(pinkn))


env = GridEnv()
env.drawGrid()

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        env.update()