import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random
# import math


class Agent:
    def step(self, observation, reward, done, info):
        pass

    def act(self, observation, random_prob=0):
        pass


class SimpleAgentCartPole(Agent):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def act(self, observation, random_prob=0):
        random_part = np.random.randn() * 0.05
        action_cont = observation[2] + random_part
        action = int(max(np.sign(action_cont), 0))

        if self.verbose:
            print(f'obs {observation[2]}')
            print(f'random: {random_part}')
            print(f'action: {action_cont}')
            print()

        return action


class QAgent(Agent):
    def __init__(self):
        self.memory = []
        self.memory_size = 50
        self.learning_rate = 0.99
        self.discount_factor = 1
        self.q_dict = None


class QAgentCartPole(Agent):
    def __init__(self):
        # self.states = np.zeros(shape=(4, 4, 4, 4))
        # self.states = np.zeros(shape=(6, 6))
        self.states = np.zeros(shape=(6, 6, 3, 2))
        # self.states = np.zeros(shape=(10, 10, 10, 10))
        self.q_table = np.zeros(shape=(np.size(self.states), 2))
        # self.q_table = np.random.randn(np.size(self.states), 2)
        # self.last_state = None
        # self.last_action = None
        # self.learning_rate = 0.05
        self.learning_rate = 0.3
        self.discount_factor = 1
        # self.discount_factor = 0.97
        self.memory = []
        self.memory_size = 50

    def step(self, observation, reward, done, info, random_prob=0):
        state = self.observation_to_state(observation)

        if self.last_state:
            self.q_table[self.last_state, self.last_action] =\
                (1 - self.learning_rate) * self.q_table[self.last_state, self.last_action]\
                + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[state]))

        if np.any(self.q_table[state]):
            # print(f'ALL! state: {state}')
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.choice([0, 1])

        # action = np.argmax(self.q_table[state])

        action = int(round(action))

        self.last_action = action
        self.last_state = state

        # return int((int(np.sign(np.random.randn())) + 1) / 2)


        # adding randomness to explore state space
        if np.random.rand() < random_prob:
            # print('RANDOM')
            action = np.random.choice([0, 1])

        # print(f'obs: {observation}')

        # print(f'action: {action}')
        # print(f'reward: {reward}')
        return action

    def act(self, observation, random_prob=0):
        state = self.observation_to_state(observation)

        if np.any(self.q_table[state]):
            # print(f'ANY! state: {state}')
            action = np.argmax(self.q_table[state])
        else:
            # print(f'NONE! state: {state}')
            action = np.random.choice([0, 1])

        action = int(round(action))

        # adding randomness to explore state space
        if np.random.rand() < random_prob:
            # print('RANDOM')
            action = np.random.choice([0, 1])

        return action

    def remember(self, prev_observation, action, observation, reward, terminal):
        state = self.observation_to_state(prev_observation)
        next_state = self.observation_to_state(observation)
        self.memory.append((state, action, next_state, reward, terminal))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def learn(self, batch_size=5):
        # batch = random.choices(self.memory, k=batch_size)
        # batch = self.memory[-10:]
        # random.shuffle(batch)
        batch = [self.memory[-1]]
        for x in batch:
            (state, action, next_state, reward, terminal) = x
            self.q_table[state, action] = \
                (1 - self.learning_rate) * self.q_table[state, action] \
                + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]))

    # def observation_to_state(self, observation):
    #     state = observation
    #     state[0] = int(state[0] * 20)
    #     state[1] = int(state[1] * 5)
    #     state[2] = int(state[2] * 50)
    #     state[3] = int(state[3] * 5)
    #
    #     state = [int(max(min(x+5-1, 9), 0)) for x in state]
    #
    #     state_idx = state[0] + state[1] * 10 + state[2] * 100 + state[3] * 1000
    #     return state_idx

    # def observation_to_state(self, observation):
    #     state = observation
    #     state[0] = int(state[0] * 20)
    #     state[1] = int(state[1] * 20)
    #     state[2] = int(state[2] * 40)
    #     state[3] = int(state[3] * 20)
    #
    #     state = [int(max(min(x+5-1, 9), 0)) for x in state]
    #
    #     state_idx = state[0] + state[1] * 10 + state[2] * 100 + state[3] * 1000
    #     return state_idx

    # def observation_to_state(self, observation):
    #     state = observation
    #     state[0] = int(state[0] * 10)
    #     state[1] = int(state[1] * 10)
    #     state[2] = int(state[2] * 20)
    #     state[3] = int(state[3] * 10)
    #
    #     state = [int(max(min(x+2-1, 3), 0)) for x in state]
    #
    #     # print(state)
    #
    #     state_idx = state[0] + state[1] * 4 + state[2] * 16 + state[3] * 64
    #     return state_idx

    def observation_to_state(self, observation):
        state = []
        # state.append(int(np.digitize(observation[2], [-0.05, -0.01, 0, 0.01, 0.05])))
        state.append(int(np.digitize(observation[2], [-0.1, -0.02, 0, 0.02, 0.1])))
        state.append(int(np.digitize(observation[3], [-0.9, -0.1, 0, 0.1, 0.9])))
        # state.append(int(np.digitize(observation[3], [-0.5, -0.1, 0, 0.1, 0.5])))
        state.append(int(np.digitize(observation[0], [-0.9, 0.9])))
        state.append(int(np.digitize(observation[1], [0])))

        # print(f'state: {state}')

        state_idx = state[0] + state[1] * 6 + state[2] * 36
        return state_idx


class QAgentTicTacToe(QAgent):
    def __init__(self, player_number):
        QAgent.__init__(self)
        self.player_number = player_number
        self.q_dict = {}
        # self.q_table = np.zeros(shape=(3 ** (board_size ** 2), board_size ** 2))

    def act(self, observation, random_prob=0):
        state = self.observation_to_state(observation)

        available_actions = self.get_available_actions(state)

        state_q_values = self.get_q_values_for_state(state)
        if np.random.rand() < random_prob:
            # random available action
            action = random.choice(available_actions)
        else:
            # random available action with maximum q value
            max_q = np.max(state_q_values)
            max_q_actions = np.argwhere(state_q_values == max_q).flatten().tolist()
            max_q_actions = [a for a in max_q_actions if a in available_actions]
            try:
                action = random.choice(max_q_actions)
            except IndexError:
                print('Max q action not in available actions')
                print(f'state: {state}')
                # print('')
                action = random.choice(available_actions)

        action = int(round(action))

        # if action not in available_actions:
        #     print('ILLEGALLLLLLLLLLLLLLLLLLL')
        #     print(state)
        #     print(state_q_values)
        #     print(observation[0])
        #     print(action)

        return action

    def get_q_values_for_state(self, state):
        if state not in self.q_dict:
            # add state to dict, assign -1000 value to illegal moves
            available_actions = self.get_available_actions(state)
            state_q_values = np.zeros(shape=(len(state),))
            for i in range(len(state_q_values)):
                if i not in available_actions:
                    # state_q_values[i] = -np.inf
                    state_q_values[i] = -1000
            self.q_dict[state] = state_q_values
        return self.q_dict[state]

    def remember(self, prev_observation, action, observation, reward, terminal):
        state = self.observation_to_state(prev_observation)
        next_state = self.observation_to_state(observation)
        self.get_q_values_for_state(next_state)
        self.memory.append((state, action, next_state, reward, terminal))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def learn(self):
        batch = [self.memory[-1]]
        for x in batch:
            (state, action, next_state, reward, terminal) = x
            self.q_dict[state][action] = \
                (1 - self.learning_rate) * self.q_dict[state][action] \
                + self.learning_rate * (reward + self.discount_factor * np.max(self.q_dict[next_state]))

    def observation_to_state(self, observation):
        '''
        Returns string representation of state, eg. '201020112'
        :param observation:
        :return:
        '''
        board, player = observation
        if self.player_number == 1:
            opponent = 2
        else:
            opponent = 1

        # TODO check if necessary
        # board_perspective is board transformed in a way that current player is always identified with 1
        board_perspective = np.zeros(shape=board.shape)
        board_perspective = board_perspective + (board == self.player_number)
        board_perspective = board_perspective + (board == opponent) * 2

        board = board_perspective.reshape(-1).tolist()
        state = ''.join(map(str, map(int, board)))
        # state_id = int(state, 3)
        return state

    # def count_all_possible_states(self, board):
    #     if board is None:
    #         board = np.zeros(shape=(self.board_size, self.board_size))
    #     indices = np.argwhere(board == 0)
    #     count = 1
    #     for ind in indices:
    #         for p in [0, 1]:
    #             board_copy = board.copy()
    #             board_copy[ind[0, 0], ind[0, 1]] = p
    #             count += self.count_all_possible_states(board_copy)
    #     return count
    @staticmethod
    def get_available_actions(state):
        return [i for i, char in enumerate(state) if char == '0']


class RandomAgentTicTacToe(Agent):
    def act(self, observation, random_prob=0):
        board = observation[0]
        board = board.reshape(-1).tolist()
        legal_actions = [i for i, p in enumerate(board) if p == 0]
        return random.choice(legal_actions)


def save_agent(agent, filename='q_agent.pkl'):
    with open(filename, 'wb+') as file:
        pkl.dump(agent, file, pkl.HIGHEST_PROTOCOL)
