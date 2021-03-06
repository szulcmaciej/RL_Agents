import numpy as np
import logging


class Game:

    def __init__(self, board_size):
        self.currentPlayer = 1
        self.gameState = GameState(np.zeros(shape=(board_size, board_size), dtype=np.int), 1)
        self.actionSpace = np.zeros(shape=(board_size ** 2), dtype=np.int)
        self.pieces = {'2': 'X', '0': '-', '1': 'O'}
        self.grid_shape = (board_size, board_size)
        # self.input_shape = (2, board_size, board_size)
        self.name = 'tic_tac_toe'
        self.state_size = self.gameState.board.shape[0]
        self.action_size = len(self.actionSpace)

    def reset(self):
        self.gameState = GameState(np.zeros(shape=self.grid_shape, dtype=np.int), 1)
        self.currentPlayer = 1
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = 2 if self.currentPlayer == 1 else 1
        info = None
        return next_state, value, done, info

    def identities(self, state, actionValues):
        identities = [(state, actionValues)]

        #TODO add identities for TicTacToe (symmetry etc.)
        #
        # currentBoard = state.board
        # currentAV = actionValues
        #
        # currentBoard = np.array([
        #     currentBoard[6], currentBoard[5], currentBoard[4], currentBoard[3], currentBoard[2], currentBoard[1],
        #     currentBoard[0]
        #     , currentBoard[13], currentBoard[12], currentBoard[11], currentBoard[10], currentBoard[9], currentBoard[8],
        #     currentBoard[7]
        #     , currentBoard[20], currentBoard[19], currentBoard[18], currentBoard[17], currentBoard[16],
        #     currentBoard[15], currentBoard[14]
        #     , currentBoard[27], currentBoard[26], currentBoard[25], currentBoard[24], currentBoard[23],
        #     currentBoard[22], currentBoard[21]
        #     , currentBoard[34], currentBoard[33], currentBoard[32], currentBoard[31], currentBoard[30],
        #     currentBoard[29], currentBoard[28]
        #     , currentBoard[41], currentBoard[40], currentBoard[39], currentBoard[38], currentBoard[37],
        #     currentBoard[36], currentBoard[35]
        # ])
        #
        # currentAV = np.array([
        #     currentAV[6], currentAV[5], currentAV[4], currentAV[3], currentAV[2], currentAV[1], currentAV[0]
        #     , currentAV[13], currentAV[12], currentAV[11], currentAV[10], currentAV[9], currentAV[8], currentAV[7]
        #     , currentAV[20], currentAV[19], currentAV[18], currentAV[17], currentAV[16], currentAV[15], currentAV[14]
        #     , currentAV[27], currentAV[26], currentAV[25], currentAV[24], currentAV[23], currentAV[22], currentAV[21]
        #     , currentAV[34], currentAV[33], currentAV[32], currentAV[31], currentAV[30], currentAV[29], currentAV[28]
        #     , currentAV[41], currentAV[40], currentAV[39], currentAV[38], currentAV[37], currentAV[36], currentAV[35]
        # ])
        #
        # identities.append((GameState(currentBoard, state.playerTurn), currentAV))
        #
        return identities


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {'2': 'X', '0': '-', '1': 'O'}
        self.playerTurn = playerTurn
        # self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedActions(self):
        flattened = self.board.flatten()
        allowed = np.argwhere(flattened == 0)
        return list(allowed.flatten())

    # def _binary(self):
    #     currentplayer_position = np.zeros(len(self.board), dtype=np.int)
    #     currentplayer_position[self.board == self.playerTurn] = 1
    #
    #     other_position = np.zeros(len(self.board), dtype=np.int)
    #     other_position[self.board == -self.playerTurn] = 1
    #
    #     position = np.append(currentplayer_position, other_position)
    #
    #     return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(shape=self.board.shape, dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(shape=self.board.shape, dtype=np.int)
        other_position[self.board == 2] = 1

        position = np.append(player1_position, other_position)
        id = ''.join(map(str, position))
        return id

    def _checkForEndGame(self):
        mask = self.board == 1
        player1_won = mask.all(0).any() | mask.all(1).any()
        player1_won |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()

        mask = self.board == 2
        player2_won = mask.all(0).any() | mask.all(1).any()
        player2_won |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()

        draw = np.count_nonzero(self.board == 0) == 0
        return player1_won or player2_won or draw

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        mask = self.board == 1
        player1_won = mask.all(0).any() | mask.all(1).any()
        player1_won |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()

        mask = self.board == 2
        player2_won = mask.all(0).any() | mask.all(1).any()
        player2_won |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()


        # TODO check if it works
        if player1_won:
            return -1, -1, 1

        return 0, 0, 0

    def _getScore(self):
        tmp = self.value
        return tmp[1], tmp[2]

    def takeAction(self, action):
        newBoard = np.array(self.board)
        action_board_pos = divmod(action, newBoard.shape[0])
        newBoard[action_board_pos[0], action_board_pos[1]] = self.playerTurn

        new_player_turn = 2 if self.playerTurn == 1 else 1

        newState = GameState(newBoard, new_player_turn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return newState, value, done

    def render(self, logger):
        for r in range(6):
            logger.info([self.pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]])
        logger.info('--------------')