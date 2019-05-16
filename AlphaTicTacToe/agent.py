import random

import tensorflow as tf
import functools

import numpy as np
import matplotlib.pyplot as plt
import mcts as mc
import config


class User:
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        action = int(input('Enter your chosen action: '))
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return action, pi, value, NN_value


class AlphaTicTacToeAgent:
    def __init__(self, name, board_size, mcts_simulations, cpuct, model):
        self.MCTSsimulations = mcts_simulations
        self.tf_model = None
        self.def_tree_iterations = 100
        self.memory = []
        self.memory_size = 10000
        self.player_number = 1
        self.opponent_number = 1 if self.player_number == 2 else 2
        self.board_size = board_size
        self.mcts = None
        self.model = model
        self.action_size = board_size ** 2
        self.cpuct = cpuct
        self.name = name

    def step(self, game_state, player_number):
        pass

    # MCTS simulation
    def simulate(self):
        # lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
        # self.mcts.root.state.render(lg.logger_mcts)
        # lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
        # leaf.state.render(lg.logger_mcts)

        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def get_preds(self, state):
        # predict the leaf
        inputToModel = np.array([self.convert_board_state_for_nn(state)])

        preds = self.model.predict(inputToModel)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]

        logits = logits_array[0]

        allowedActions = self.allowed_actions(state)

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)  ###put this just before the for?

        return value, probs, allowedActions

    def allowed_actions(self, numpy_board):
        flattened = numpy_board.flatten()
        allowed = np.argwhere(flattened == 0)
        return list(allowed.flatten())

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):

        # lg.logger_mcts.info('------EVALUATING LEAF------')

        if done == 0:

            value, probs, allowedActions = self.get_preds(leaf.state)
            # lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction(action)
                if newState.id not in self.mcts.tree:
                    node = mc.Node(newState)
                    self.mcts.addNode(node)
                    # lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
                else:
                    node = self.mcts.tree[newState.id]
                    # lg.logger_mcts.info('existing node...%s...', node.id)

                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))

        # else:
        #     lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

        return ((value, breadcrumbs))

    def act(self, state, tau):

        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation
        for sim in range(self.MCTSsimulations):
            # lg.logger_mcts.info('***************************')
            # lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            # lg.logger_mcts.info('***************************')
            self.simulate()

        #### get action values
        pi, values = self.getAV(1)

        ####pick the action
        action, value = self.chooseAction(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = -self.get_preds(nextState)[0]

        # lg.logger_mcts.info('ACTION VALUES...%s', pi)
        # lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        # lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        # lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

        return action, pi, value, NN_value

    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory):
        # lg.logger_mcts.info('******RETRAINING MODEL******')

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
                , 'policy_head': np.array([row['AV'] for row in minibatch])}

            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=config.BATCH_SIZE)
            # lg.logger_mcts.info('NEW LOSS %s', fit.history)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

        # plt.plot(self.train_overall_loss, 'k')
        # plt.plot(self.train_value_loss, 'k:')
        # plt.plot(self.train_policy_loss, 'k--')
        #
        # plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
        #
        # display.clear_output(wait=True)
        # display.display(pl.gcf())
        # pl.gcf().clear()
        # time.sleep(1.0)
        #
        # print('\n')
        # self.model.printWeightAverages()

    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds

    def buildMCTS(self, state):
        # lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpuct)

    def changeRootMCTS(self, state):
        # lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]

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

# def main():
#     model =
#     agent = AlphaTicTacToeAgent('agent_no_1', 3, 50, 1)
#     bs = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 2]])
#     print(bs)
#     nn_board = agent.convert_board_state_for_nn(bs)
#
#     print(nn_board)
#
#
# if __name__ == '__main__':
#     main()
