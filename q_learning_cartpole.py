import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random

from agents import QAgentCartPole


def play(games, env, agent, random_scale=0.4, verbose=True, display=False, plot=False):

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
            if display:
                env.render()

            random_prob = random_scale * ((games - i) / games) ** 4
            action = agent.act(prev_observation, random_prob=random_prob)
            observation, reward, done, info = env.step(action)
            if done and steps < 199:
                reward = -10
            agent.remember(prev_observation, action, observation, reward, done)
            agent.learn()
            prev_observation = observation

            cum_reward += reward
            steps += 1

        results.append(steps)
        if verbose:
            if steps == 200:
                print(f'game {i} finished after {steps} steps' + '*'.rjust(steps // 2, '\u25A0'))
            else:
                print(f'game {i} finished after {steps} steps' + '*'.rjust(steps // 2, '|'))

    env.close()

    if plot:
        plot_results(results)

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


def show(env, agent):
    play(2, env, agent, 0, display=True, plot=False)


def compare_random_scales():
    scales = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 2]
    env = gym.make('CartPole-v0')
    games = 500
    last_n_results_proc = 10
    repeats = 10

    for scale in scales:
        scores = []
        for repeat in range(repeats):
            agent = QAgentCartPole()
            results = play(games, env, agent, random_scale=scale, verbose=False)
            no_of_results_to_score = int(games * last_n_results_proc / 100)
            scores.append(np.mean(results[no_of_results_to_score:]))
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)

        print(f'random_scale: {scale} scores_mean: {scores_mean} scores_std: {scores_std}')


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = QAgentCartPole()
    # results = play(1000, env, agent, verbose=False, random_scale=0.1)
    results = play(300, env, agent, display=False, random_scale=0.5)

    show(env, agent)
    # compare_random_scales()
