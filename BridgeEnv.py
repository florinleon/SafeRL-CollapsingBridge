import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pygame
import random
from itertools import filterfalse


class BridgeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=8, render_mode=None, phase="train"):
        #Grid-based RL environment with sparse rewards, partial observability,
        #irreversible transitions (collapsing bridges), and task variation
        super(BridgeEnv, self).__init__()

        self.nondeterministic = False
        self.impossible = False

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.phase = phase  # "train" or "test"

        self.usable_top = 1
        self.usable_bottom = 6
        self.usable_left = 1
        self.usable_right = 6

        # Define terrain types
        self.cell_wall   = 0
        self.cell_floor  = 1
        self.cell_water  = 2
        self.cell_bridge = 3
        self.cell_time   = 4
        self.cell_object = 5
        self.cell_goal   = 6

        # Action space: movement only (no explicit pick/drop); interaction is automatic via position
        self.action_space = spaces.Discrete(4)  # 0=left, 1=right, 2=up, 3=down

        # Observation encodes local spatial context (partial observability), agent location, time, and object status
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(low=0, high=6, shape=(3, 3), dtype=np.int32),
            "agent_pos": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
            "time_indicator": spaces.Discrete(2),
            "has_object": spaces.Discrete(2)
        })

        # Sparse reward structure favors efficiency and penalizes failure
        self.step_reward = -1
        self.slight_penalty = -1
        self.termination_failure = -1000
        self.termination_success = 1000

        self.max_steps = 101

        self.agent_start_pos = (1, 1)
        self.goal_pos = (6, 1)

        # Static layout per training or test phase
        self.water_cells_train = [(1,3), (1,4), (4,3), (4,4), (5,3), (5,4), (6,3), (6,4)]
        self.bridge_cells_train = [(2,3), (2,4), (3,3), (3,4)]
        self.water_cells_test = [(1,3), (1,4), (2,3), (2,4), (5,3), (5,4), (6,3), (6,4)]
        self.bridge_cells_test = [(3, 3), (3, 4), (4, 3), (4, 4)]

        # Object location determined by time of day (randomized at reset)
        self.object_loc_morning = (1, 6)
        self.object_loc_evening = (6, 6)

        # Environment state
        self.agent_pos = None
        self.grid = None
        self.steps = 0
        self.has_object = False
        self.object_pos = None
        self.last_bridge_cell = None

        self.window = None
        self.cell_size = 50


    def _init_grid(self):
        grid = np.full((self.grid_size, self.grid_size), self.cell_floor, dtype=np.int32)
        grid[0, :] = self.cell_wall
        grid[self.grid_size - 1, :] = self.cell_wall
        grid[:, 0] = self.cell_wall
        grid[:, self.grid_size - 1] = self.cell_wall
        grid[0, 0] = self.cell_time

        if self.phase == "train":
            self.water_cells = self.water_cells_train
            bridge_cells = self.bridge_cells_train
        else:
            self.water_cells = self.water_cells_test
            bridge_cells = self.bridge_cells_test

        for (r, c) in self.water_cells:
            grid[r, c] = self.cell_water
        for (r, c) in bridge_cells:
            grid[r, c] = self.cell_bridge

        gr, gc = self.goal_pos
        grid[gr, gc] = self.cell_goal

        if self.impossible:
            grid[2, 1] = self.cell_wall

        return grid


    def reset(self, seed=None, options=None):
        self.steps = 0
        self.has_object = False
        self.agent_pos = self.agent_start_pos
        self.grid = self._init_grid()
        # Randomize time-of-day each episode to vary object location (adds task diversity)
        self.time_of_day = random.choice([0, 1])
        self.object_pos = self.object_loc_morning if self.time_of_day == 0 else self.object_loc_evening
        self.last_bridge_cell = None
        return self._get_obs(), {}


    def _get_obs(self):
        # Extract local 3x3 view around agent (partial observability)
        r, c = self.agent_pos
        local = np.zeros((3, 3), dtype=np.int32)
        for i in range(-1, 2):
            for j in range(-1, 2):
                rr, cc = r + i, c + j
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    local[i + 1, j + 1] = self.grid[rr, cc]
                    if self.object_pos is not None and (rr, cc) == self.object_pos:
                        local[i + 1, j + 1] = self.cell_object
                else:
                    local[i + 1, j + 1] = self.cell_wall
        return {
            "local_grid": local,
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "time_indicator": self.time_of_day,
            "has_object": int(self.has_object)
        }


    def step(self, action):
        action = int(action)

        # Collapse the last bridge cell after agent leaves it
        if self.last_bridge_cell is not None:
            r_bridge, c_bridge = self.last_bridge_cell
            self.grid[r_bridge, c_bridge] = self.cell_water
            self.last_bridge_cell = None

        self.steps += 1
        reward = self.step_reward
        done = False
        info = {}

        r, c = self.agent_pos
        delta = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        dr, dc = delta.get(action, (0, 0))

        move_distance = 1 if not self.nondeterministic or random.random() < 0.8 else 2
        nr, nc = r + dr * move_distance, c + dc * move_distance

        new_pos = self.agent_pos

        # Determine effects of attempted movement
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            cell_type = self.grid[nr, nc]

            if cell_type == self.cell_wall:
                reward += self.slight_penalty
            elif cell_type == self.cell_water:
                self.agent_pos = (nr, nc)
                reward = self.termination_failure
                done = True
                return self._get_obs(), reward, done, False, info
            else:
                new_pos = (nr, nc)
        else:
            reward += self.slight_penalty  # out-of-bounds

        self.agent_pos = new_pos
        nr, nc = self.agent_pos

        if self.grid[nr, nc] == self.cell_bridge:
            self.last_bridge_cell = (nr, nc)

        # Pickup logic
        if not self.has_object and self.object_pos is not None and self.agent_pos == self.object_pos:
            self.has_object = True
            self.object_pos = None

        # Successful delivery logic
        if self.has_object and self.agent_pos == self.goal_pos:
            reward += self.termination_success
            done = True
            self.has_object = False

        if self.steps >= self.max_steps:
            reward = self.termination_failure
            done = True

        return self._get_obs(), reward, done, False, info


    def render(self, mode="human"):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
                pygame.display.set_caption("Collapsing Bridge Environment")
            self.window.fill((255, 255, 255))

            colors = {
                self.cell_wall: (50, 50, 50),
                self.cell_floor: (220, 220, 220),
                self.cell_water: (0, 100, 255),
                self.cell_bridge: (139, 69, 19),
                self.cell_time: (255, 165, 0),
                self.cell_object: (255, 0, 0),
                self.cell_goal: (0, 255, 255)
            }

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell = self.grid[i, j]
                    rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.window, colors.get(cell, (255, 255, 255)), rect)
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 1)
                    if i == 0 and j == 0:
                        font = pygame.font.SysFont(None, 24)
                        img = font.render("T", True, (0, 0, 0))
                        self.window.blit(img, (j * self.cell_size + 5, i * self.cell_size + 5))

            if self.object_pos is not None:
                orow, ocol = self.object_pos
                center = (ocol * self.cell_size + self.cell_size // 2, orow * self.cell_size + self.cell_size // 2)
                pygame.draw.circle(self.window, colors[self.cell_object], center, self.cell_size // 4)

            ar, ac = self.agent_pos
            center = (ac * self.cell_size + self.cell_size // 2, ar * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(self.window, (0, 255, 0), center, self.cell_size // 3)

            pygame.display.flip()
            pygame.time.wait(500)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        else:
            obs = self._get_obs()
            print("Observation:", obs)


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


# Register with Gym
register(id="BridgeEnv-v0", entry_point="BridgeEnv:BridgeEnv")
