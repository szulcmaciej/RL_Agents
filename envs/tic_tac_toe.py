"""
My implementation of Tic Tac Toe for OpenAI gym
"""

import math
import gym
from gym.envs.registration import registry, register, spec
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
import time


class TicTacToeEnv(gym.Env):
    # TODO change description
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    rewards = {
        'win': 10.0,
        # 'lose': -1.0,
        'draw': 1.0,
        # 'draw': 0.0,
        'illegal': -100.0,
        'playing': 0.0
    }

    def __init__(self, render_sleep_time=0.3):
        self.board_size = 4

        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Discrete(self.board_size ** 2 * 3 * 2)

        self.viewer = None
        self.state = None
        
        self.render_x_positions = None
        self.render_y_positions = None
        self.render_sleep_time = render_sleep_time
        
        # TODO is it needed?
        self.steps_beyond_done = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.state is not None
        state = self.state
        board, player = state
        info = {}

        # action_board_pos = (action // self.board_size, action % self.board_size)
        action_board_pos = divmod(action, self.board_size) #equal to line above
        # print(action_board_pos)
        # check if move legal
        if board[action_board_pos[0], action_board_pos[1]] != 0:
            # illegal move
            print('ILLEGAL MOVE')
            reward = self.rewards['illegal']
            done = False
            if player == 1:
                player = 2
            elif player == 2:
                player = 1
        else:
            # legal move
            # print('LEGAL MOVE')
            # update board
            board[action_board_pos[0], action_board_pos[1]] = player

            # check if player won
            mask = board == player
            player_won = mask.all(0).any() | mask.all(1).any()
            player_won |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()
            draw = np.count_nonzero(board == 0) == 0

            if player_won:
                reward = self.rewards['win']
                info['result'] = player
            elif draw:
                reward = self.rewards['draw']
                info['result'] = 0
            else:
                # still playing, change player
                reward = self.rewards['playing']
                if player == 1:
                    player = 2
                elif player == 2:
                    player = 1

            done = player_won or draw

        self.state = [board, player]





        return self.state, reward, done, info

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = [np.zeros(shape=(self.board_size, self.board_size)), 1]
        self.steps_beyond_done = None
        if self.viewer:
            self.viewer.geoms.clear()
        return self.state

    def render(self, mode='human'):
        cell_size = 100
        cell_padding = 1
        sign_relative_size = 0.7

        screen_width = cell_size * self.board_size + cell_padding * (self.board_size - 1)
        screen_height = cell_size * self.board_size + cell_padding * (self.board_size - 1)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if self.state is None:
            return None

        if self.render_x_positions is None or self.render_y_positions is None:
            self.render_x_positions = np.zeros(shape=(self.board_size, self.board_size))
            self.render_y_positions = np.zeros(shape=(self.board_size, self.board_size))
            center = (screen_width / 2, screen_height / 2)
            for i in range(self.board_size):
                for j in range(self.board_size):
                    self.render_x_positions[i, j] = i * (cell_size + cell_padding) + cell_size / 2
                    self.render_y_positions[i, j] = j * (cell_size + cell_padding) + cell_size / 2
                    # self.render_x_positions[i, j] = center[0] + (i + 2 - self.board_size + ((self.board_size + 1)%2)/2) * (cell_size + cell_padding)
                    # self.render_y_positions[i, j] = center[1] + (j + 2 - self.board_size + ((self.board_size + 1)%2)/2) * (cell_size + cell_padding)

        # if no board, draw board
        if len(self.viewer.geoms) == 0:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    cell = rendering.FilledPolygon(
                        [(self.render_x_positions[i, j] - cell_size / 2, self.render_y_positions[i, j] - cell_size / 2),
                         (self.render_x_positions[i, j] - cell_size / 2, self.render_y_positions[i, j] + cell_size / 2),
                         (self.render_x_positions[i, j] + cell_size / 2, self.render_y_positions[i, j] + cell_size / 2),
                         (self.render_x_positions[i, j] + cell_size / 2, self.render_y_positions[i, j] - cell_size / 2)])
                    # cell.set_color(0.5, 0.5, 0.5)
                    self.viewer.add_geom(cell)

        # fill board with game state
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.state[0][i, j] == 1:
                    circle = rendering.make_circle(cell_size * sign_relative_size / 2, filled=True)
                    circle_trans = rendering.Transform(translation=(self.render_x_positions[i,j], self.render_y_positions[i, j]))
                    circle.add_attr(circle_trans)
                    circle.set_color(0, 0, 1)
                    self.viewer.add_geom(circle)
                if self.state[0][i, j] == 2:
                    circle = rendering.make_circle(cell_size * sign_relative_size / 2, filled=True)
                    circle_trans = rendering.Transform(translation=(self.render_x_positions[i,j], self.render_y_positions[i, j]))
                    circle.add_attr(circle_trans)
                    circle.set_color(1, 0, 0)
                    self.viewer.add_geom(circle)

        time.sleep(self.render_sleep_time)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # console text render below

    # def render(self, mode='human'):
    #     print()
    #     board_str = str(self.state[0]).replace('0.', ' ').replace('1.', 'O').replace('2.', 'X')
    #     print(f'Player {self.state[1]}')
    #     print(board_str)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def register_env():
    register(
        id='TicTacToe-v0',
        entry_point='envs.tic_tac_toe:TicTacToeEnv',
        max_episode_steps=200,
        reward_threshold=1.0,
    )
