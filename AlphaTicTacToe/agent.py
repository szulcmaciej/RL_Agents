import tensorflow as tf
import functools

import numpy as np
import matplotlib.pyplot as plt


class AlphaTicTacToeAgent:
    def __init__(self, board_size):
        self.tf_model = None
        self.def_tree_iterations = 100
        self.memory = []
        self.memory_size = 10000
        self.player_number = 1
        self.opponent_number = 1 if self.player_number == 2 else 2
        self.board_size = board_size

    def step(self, game_state, player_number):
        pass

    def mcts_search(self, game_state):
        pass

    def train_model(self):
        pass

    def create_tf_model(self):
        pass

    def convert_board_state_for_nn(self, numpy_board):
        numpy_board = np.array(numpy_board)
        nn_board = np.zeros(shape=(3, numpy_board.shape[0], numpy_board.shape[1]))
        nn_board[0, :, :] = (numpy_board == 1)
        nn_board[1, :, :] = (numpy_board == 2)
        nn_board[2, :, :] = self.player_number - 1

        return nn_board


def main():
    agent = AlphaTicTacToeAgent(3)
    bs = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 2]])
    print(bs)
    nn_board = agent.convert_board_state_for_nn(bs)

    print(nn_board)


if __name__ == '__main__':
    main()
