import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


class Agent:
    def step(self, observation, reward, done, info):
        pass


class SimpleAgent(Agent):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def step(self, observation, reward, done, info):
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
        self.states = np.zeros(shape=(4, 4, 4, 4))
        # self.states = np.zeros(shape=(10, 10, 10, 10))
        self.q_table = np.zeros(shape=(np.size(self.states), 2))
        self.last_state = None
        self.last_action = None
        self.learning_rate = 0.1
        # self.learning_rate = 1
        self.discount_factor = 0.97

    # TODO split into 3 functions (act, remember/save, learn)
    # TODO add negative reward on termination of the game
    def step(self, observation, reward, done, info, random_prob=0):
        state = self.observation_to_state(observation)

        if self.last_state:
            self.q_table[self.last_state, self.last_action] =\
                (1 - self.learning_rate) * self.q_table[self.last_state, self.last_action]\
                + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[state]))

        if np.all(self.q_table[state]):
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
    #     state[0] = int(state[0] * 10)
    #     state[1] = int(state[1] * 10)
    #     state[2] = int(state[2] * 20)
    #     state[3] = int(state[3] * 10)
    #
    #     state = [int(max(min(x+5-1, 9), 0)) for x in state]
    #
    #     state_idx = state[0] + state[1] * 10 + state[2] * 100 + state[3] * 1000
    #     return state_idx

    def observation_to_state(self, observation):
        state = observation
        state[0] = int(state[0] * 10)
        state[1] = int(state[1] * 10)
        state[2] = int(state[2] * 20)
        state[3] = int(state[3] * 10)

        state = [int(max(min(x+2-1, 3), 0)) for x in state]

        # print(state)

        state_idx = state[0] + state[1] * 4 + state[2] * 16 + state[3] * 64
        return state_idx


def play(games, random_scale=0.4):

    # with open('q_agent.pkl', 'rb') as f:
    #     agent = pkl.load(f)

    results = []
    actions = []

    for i in range(games):
        observation = env.reset()
        done = False
        reward = 0
        info = {}
        steps = 0
        cum_reward = 0
        while not done:
            prev_obs = observation

            # env.render()

            # action = agent.step(observation, reward, done, info, random_prob=0)
            # random_prob = 0.3 * ((games - i) / games) ** 4
            random_prob = random_scale * ((games - i) / games) ** 4
            # random_prob = 0
            action = agent.step(observation, reward, done, info, random_prob=random_prob)
            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            cum_reward += reward
            steps += 1

            actions.append(action)

        results.append(steps)
        # print(f'game {i} finished after {steps} steps' + '*'.rjust(steps, '|'))

    env.close()

    plot_results(results)

    # print(f'ones: {sum(actions)}/{len(actions)}')

    return results


def plot_results(results):
    plt.plot(np.unique(range(len(results))), np.poly1d(np.polyfit(range(len(results)), results, 1))(np.unique(range(len(results)))), color='r')
    plt.scatter(range(len(results)), results, marker='.')
    plt.grid(which='both')
    plt.show()


def save_agent(filename='q_agent.pkl'):
    with open(filename, 'wb+') as file:
        pkl.dump(agent, file, pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = QAgent()
    results = play(10000)
