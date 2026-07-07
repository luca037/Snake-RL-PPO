import pygame
import random
from enum import Enum
from collections import namedtuple

import numpy as np
import math

import gymnasium as gym
from gymnasium import spaces

pygame.init()
font = pygame.font.SysFont('arial', 24)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

### RGB COLORS ####
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

HEAD_COLOR1 = (0, 100, 0)
HEAD_COLOR2 = (0, 150, 0)
TAIL_COLOR1 = (100, 0, 0)
TAIL_COLOR2 = (100, 60, 0)
BACKGROUND = (44, 178, 169)

### GAME SETTINGS ####
BLOCK_SIZE = 20
SPEED = 20
BOARD_W = 10
BOARD_H = 10


class SnakeGame:
    """Core snake game logic, decoupled from rendering."""

    def __init__(self, w=200, h=200, gui=True):
        # Window width and height.
        self.w = w
        self.h = h

        self.max_len = (w // BLOCK_SIZE) * (h // BLOCK_SIZE)

        # If gui is active or not.
        self.gui = gui

        # Init display.
        if self.gui:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Score: 0')

        # Set clock.
        self.clock = pygame.time.Clock()

        # Init game.
        self.reset()

    def reset(self):
        # First direction is right.
        self.direction = Direction.RIGHT

        # Place the head position.
        self.head = Point(self.w / 2, self.h / 2)

        # Place the entire snake.
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        # Init score.
        self.score = 0

        # Init food.
        self.food = Point(-1, -1)
        self._place_food()

        # Track distance for potential-based reward shaping.
        self.prev_dist = self._manhattan_dist()

        # Init step counter.
        self.step_counter = 0

    def _manhattan_dist(self):
        # Manhattan distance from head to food in grid units.
        return abs(self.head.x - self.food.x) / BLOCK_SIZE + abs(self.head.y - self.food.y) / BLOCK_SIZE

    def _place_food(self):
        # Generate random coordinates.
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        # Loop until position is outside snake body.
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.step_counter += 1

        # Manage quit game.
        if self.gui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Move the snake.
        self.move(action)
        self.snake.insert(0, self.head)

        # Check if game over.
        reward = 0.0
        game_over = False
        if self.is_collision() or self.step_counter > 100 * len(self.snake):
            game_over = True
            reward = -1.0
            self.snake.pop()
            return reward, game_over, self.score

        # Place new food or just move.
        if self.head == self.food:
            self.score += 1
            reward = 1.0
            if len(self.snake) == self.max_len:
                # The snake filled the entire board.
                game_over = True
                reward = 100.0
            else:
                self._place_food()
                self.prev_dist = self._manhattan_dist()
        else:
            self.snake.pop()

            # Potential-based reward shaping: encourage moving toward food.
            curr_dist = self._manhattan_dist()
            if curr_dist < self.prev_dist:
                reward += 0.005
            elif curr_dist > self.prev_dist:
                reward -= 0.005
            self.prev_dist = curr_dist

            # Small step penalty to discourage looping.
            reward -= 0.001

        # Update ui and clock.
        if self.gui:
            self._update_ui()
            self.clock.tick(SPEED)

        # Return game over and score.
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Hits boundary.
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself.
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BACKGROUND)

        # Color head and tail.
        pygame.draw.rect(self.display, HEAD_COLOR1, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, HEAD_COLOR2, pygame.Rect(self.head.x + 4, self.head.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        pygame.draw.rect(self.display, TAIL_COLOR1, pygame.Rect(self.snake[-1].x, self.snake[-1].y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, TAIL_COLOR2, pygame.Rect(self.snake[-1].x + 4, self.snake[-1].y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

        for pt in self.snake[1:-1]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        if len(self.snake) < self.max_len:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.set_caption(f'S: {str(self.score)}')
        pygame.display.flip()

    def move(self, action, perform=True):
        # Action parsing: [straight, right, left].

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]

        # Get index of current direction.
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        x = self.head.x
        y = self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        # Update direction and head if necessary.
        if perform:
            self.direction = new_dir
            self.head = Point(x, y)

        return Point(x, y)


class GymSnakeEnv(gym.Env):
    """Gymnasium wrapper with multi-channel observation space."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Channel indices for the observation tensor.
    CH_WALLS = 0
    CH_HEAD = 1
    CH_BODY = 2
    CH_FOOD = 3
    CH_DIR = 4

    NUM_CHANNELS = 5

    def __init__(self, frame_rows=12, frame_cols=12, gui=False):
        super().__init__()
        self.frame_rows = frame_rows
        self.frame_cols = frame_cols

        self.game = SnakeGame(gui=gui)
        self.action_space = spaces.Discrete(3)

        # Multi-channel observation: (C, H, W) for SB3 CnnPolicy.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.NUM_CHANNELS, frame_rows, frame_cols),
            dtype=np.float32
        )

    def _get_obs(self):
        # Build a multi-channel observation tensor.
        obs = np.zeros((self.NUM_CHANNELS, self.frame_rows, self.frame_cols), dtype=np.float32)

        # Channel 0: Walls (border cells = 1).
        obs[self.CH_WALLS, 0, :] = 1.0
        obs[self.CH_WALLS, self.frame_rows - 1, :] = 1.0
        obs[self.CH_WALLS, :, 0] = 1.0
        obs[self.CH_WALLS, :, self.frame_cols - 1] = 1.0

        # Channel 2: Body with ordering (closer to head = higher value).
        snake_len = len(self.game.snake)
        for idx, point in enumerate(self.game.snake[1:]):
            i = int(point.y // BLOCK_SIZE) + 1
            j = int(point.x // BLOCK_SIZE) + 1
            if 0 <= i < self.frame_rows and 0 <= j < self.frame_cols:
                # Linearly decay from 1.0 (near head) to ~0.1 (tail).
                obs[self.CH_BODY, i, j] = 1.0 - (0.9 * idx / max(snake_len - 1, 1))

        # Channel 1: Head position.
        hi = int(self.game.head.y // BLOCK_SIZE) + 1
        hj = int(self.game.head.x // BLOCK_SIZE) + 1
        if 0 <= hi < self.frame_rows and 0 <= hj < self.frame_cols:
            obs[self.CH_HEAD, hi, hj] = 1.0

        # Channel 3: Food position.
        fi = int(self.game.food.y // BLOCK_SIZE) + 1
        fj = int(self.game.food.x // BLOCK_SIZE) + 1
        if 0 <= fi < self.frame_rows and 0 <= fj < self.frame_cols:
            obs[self.CH_FOOD, fi, fj] = 1.0

        # Channel 4: Direction encoding (fill entire channel with direction ID).
        direction_map = {
            Direction.UP: 0.25,
            Direction.RIGHT: 0.50,
            Direction.DOWN: 0.75,
            Direction.LEFT: 1.00,
        }
        obs[self.CH_DIR, :, :] = direction_map.get(self.game.direction, 0.0)

        return obs

    def step(self, action):
        final_move = [0, 0, 0]
        final_move[action] = 1

        reward, gameover, score = self.game.play_step(final_move)
        observation = self._get_obs()

        info = {"score": score}
        return observation, float(reward), bool(gameover), False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}


def snake_make_env(frame_rows=12, frame_cols=12, gui=False):
    """Utility to create the wrapped environment for SB3 vectorization."""
    env = GymSnakeEnv(frame_rows=frame_rows, frame_cols=frame_cols, gui=gui)
    return env
