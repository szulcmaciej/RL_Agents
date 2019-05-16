import numpy as np


import config


class Node:
    def __init__(self, state):
        self.state = state
        self.id = self.state.id
        self.player_turn = state.playerTurn
        self.edges = []

    def isLeaf(self):
        return len(self.edges) == 0


class Edge:
    def __init__(self, in_node, out_node, prior, action):
        self.inNode = in_node
        self.outNode = out_node
        self.action = action
        self.playerTurn = self.inNode.state.playerTurn

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior
        }


class MCTS:
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        # lg.logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = 0

        while not currentNode.isLeaf():

            # lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)

            maxQU = -99999

            if currentNode == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):

                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                # lg.logger_mcts.info(
                #     'action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
                #     , action, action % 7, edge.stats['N'], np.round(edge.stats['P'], 6), np.round(nu[idx], 6),
                #     ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx])
                #     , np.round(edge.stats['W'], 6), np.round(Q, 6), np.round(U, 6), np.round(Q + U, 6))

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            # lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

            newState, value, done = currentNode.state.takeAction(
                simulationAction)  # the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        # lg.logger_mcts.info('DONE...%d', done)

        return currentNode, value, done, breadcrumbs

    def backFill(self, leaf, value, breadcrumbs):
        # lg.logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.state.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            # lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
            #                     , value * direction
            #                     , playerTurn
            #                     , edge.stats['N']
            #                     , edge.stats['W']
            #                     , edge.stats['Q']
            #                     )

            # edge.outNode.state.render(lg.logger_mcts)

    def add_node(self, node):
        self.tree[node.id] = node
