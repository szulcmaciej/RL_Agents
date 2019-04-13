import gym
import numpy as np
import matplotlib.pyplot as plt


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
        self.states = np.zeros(shape=(10, 10, 10, 10))
        self.q_table = np.zeros(shape=(np.size(self.states), 2))
        self.last_state = None
        self.last_action = None

    def step(self, observation, reward, done, info, random_prob=0):
        state = self.observation_to_state()

        if self.last_state:
            self.q_table[self.last_state, self.last_action] += reward + np.max(self.q_table[state])

        action = np.argmax(self.q_table[state])
        self.last_action = action
        self.last_state = state

        # return int((int(np.sign(np.random.randn())) + 1) / 2)


        # adding randomness to explore state space
        if np.random.rand() < random_prob:
            action = abs(action - 1)


        # print(f'action: {action}')
        print(f'reward: {reward}')
        return action

    def observation_to_state(self):
        state = observation
        state[0] = int(state[0] * 20)
        state[1] = int(state[1] * 5)
        state[2] = int(state[2] * 50)
        state[3] = int(state[3] * 5)

        state = [int(max(min(x+5-1, 9), 0)) for x in state]

        state_idx = state[0] + state[1] * 10 + state[2] * 100 + state[3] * 1000
        return state_idx



env = gym.make('CartPole-v0')

agent = QAgent()
results = []

for i in range(2):
    observation = env.reset()
    done = False
    reward = 0
    info = {}
    steps = 0
    while not done:
        prev_obs = observation

        env.render()
        action = agent.step(observation, reward, done, info, random_prob=0.2)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        steps += 1


        # print(observation[:2])
        # print(observation[0] - prev_obs[0])
        # print(observation[1])
        # print((observation[0] - prev_obs[0]) / observation[1])
        # print()

        # print(observation)
    results.append(steps)
    print(f'game {i} finished after {steps} steps')

env.close()

# plt.plot(results)
# plt.show()
