"""
My implementation of Tic Tac Toe for OpenAI gym
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


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
        'win': 1.0,
        'lose': -1.0,
        'draw': 0.0,
        'illegal': -100.0,
        'playing': 0.0
    }

    def __init__(self):
        self.board_size = 3

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds

        self.action_space = spaces.Discrete(9)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = spaces.Discrete(self.board_size ** 2 * 3 * 2)

        #TODO usun to
        # self.seed()

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    # TODO usun to
    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        assert self.state is not None
        state = self.state
        board, player = state

        action_board_pos = (action // self.board_size, action % self.board_size)
        print(action_board_pos)
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
            elif draw:
                reward = self.rewards['draw']
            else:
                # still playing, change player
                reward = self.rewards['playing']
                if player == 1:
                    player = 2
                elif player == 2:
                    player = 1

            done = player_won or draw

        self.state = [board, player]

        return self.state, reward, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = [np.zeros(shape=(3,3)), 1]
        self.steps_beyond_done = None
        return self.state

    # def render(self, mode='human'):
    #     screen_width = 600
    #     screen_height = 400
    #
    #     world_width = self.x_threshold * 2
    #     scale = screen_width / world_width
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * (2 * self.length)
    #     cartwidth = 50.0
    #     cartheight = 30.0
    #
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #         axleoffset = cartheight / 4.0
    #         cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    #         pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         pole.set_color(.8, .6, .4)
    #         self.poletrans = rendering.Transform(translation=(0, axleoffset))
    #         pole.add_attr(self.poletrans)
    #         pole.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole)
    #         self.axle = rendering.make_circle(polewidth / 2)
    #         self.axle.add_attr(self.poletrans)
    #         self.axle.add_attr(self.carttrans)
    #         self.axle.set_color(.5, .5, .8)
    #         self.viewer.add_geom(self.axle)
    #         self.track = rendering.Line((0, carty), (screen_width, carty))
    #         self.track.set_color(0, 0, 0)
    #         self.viewer.add_geom(self.track)
    #
    #         self._pole_geom = pole
    #
    #     if self.state is None: return None
    #
    #     # Edit the pole polygon vertex
    #     pole = self._pole_geom
    #     l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    #     pole.v = [(l, b), (l, t), (r, t), (r, b)]
    #
    #     x = self.state
    #     cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    #     self.carttrans.set_translation(cartx, carty)
    #     self.poletrans.set_rotation(-x[2])
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render(self, mode='human'):
        print()
        board_str = str(self.state[0]).replace('0.', ' ').replace('1.', 'O').replace('2.', 'X')
        print(f'Player {self.state[1]}')
        print(board_str)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None