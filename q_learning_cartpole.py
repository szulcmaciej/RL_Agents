import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random

from agents import QAgentCartPole


def play(games, random_scale=0.4, verbose=False):

    # with open('q_agent.pkl', 'rb') as f:
    #     agent = pkl.load(f)

    results = []

    for i in range(games):
        prev_observation = env.reset()
        done = False
        # reward = 0
        # info = {}
        steps = 0
        cum_reward = 0
        while not done:
            if verbose:
                env.render()

            random_prob = random_scale * ((games - i) / games) ** 4
            action = agent.act(prev_observation, random_prob=random_prob)
            observation, reward, done, info = env.step(action)
            reward = reward if not done else -10
            agent.remember(prev_observation, action, observation, reward, done)
            agent.experience_replay()
            prev_observation = observation

            cum_reward += reward
            steps += 1

        results.append(steps)
        print(f'game {i} finished after {steps} steps' + '*'.rjust(steps, '|'))

    env.close()

    plot_results(results)

    # print(f'ones: {sum(actions)}/{len(actions)}')

    return results


def plot_results(results):
    x = range(len(results))
    y = results
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='r')
    plt.scatter(x, y, marker='.')
    plt.grid(which='both')
    plt.show()


def save_agent(filename='q_agent.pkl'):
    with open(filename, 'wb+') as file:
        pkl.dump(agent, file, pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = QAgentCartPole()
    results = play(20, verbose=False, random_scale=0.1)
