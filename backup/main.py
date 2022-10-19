from gym import Env
import numpy as np
from gym import error, spaces, utils


class BlockStackEnv(Env):
    def __init__(self):
        # Action Space
        self.actions = ['up', 'right', 'down', 'left', 'open', 'close', 'grasp', 'pickup', 'drop']
        self.action_space = spaces.Discrete(9)

        self.actions_pos_dict = {up:[-1,0], down:[1,0], right:[0,-1], left:[0,1], begin:[0,0]}

        # Observation Space
        self.obs_space = [3, 3, 3]
        self.observation_space = spaces.Box(low=0, high=2, shape=self.obs_space)

        self.initial_map = np.array([np.array([np.array([0] * 3)] * 3)] * 3)
        self.grid_shape = self.initial_map.shape

        print(self.initial_map)

        self.agent_state = np.array([0, 0, 0])
        self.arm_state = 0  # 0 closed, 1 open, 2 grasp, 3 pickup, 4 drop

        # Construct Grid

    def step(self):
        print("taking step")
        action = self.actions



blockstack = BlockStackEnv()

