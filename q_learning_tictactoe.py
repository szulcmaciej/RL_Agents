import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import random

from envs.tic_tac_toe import register_env
from agents import QAgentTicTacToe, RandomAgentTicTacToe


# register_env()
# env = gym.make('TicTacToe-v0')


def play(games, env, agents, random_scale=0.4, verbose=True, display=False, plot=False, load=False, save=False):

    if load:
        with open('q_agent.pkl', 'rb') as f:
            agent = pkl.load(f)

    results = []

    for i in range(games):
        prev_observation = env.reset()
        done = False
        reward = 0
        info = {}
        steps = 0
        cum_reward = 0
        player_id = prev_observation[1]
        observation = prev_observation.copy()
        while not done:
            if display:
                env.render()

            player_id = prev_observation[1]
            agent = agents[player_id - 1]
            opponent = agents[1] if player_id == 1 else agents[0]

            # TODO fix - observation equals prev_observation in agent.remember()
            random_prob = random_scale * ((games - i) / games) ** 4
            action = agent.act(prev_observation, random_prob=random_prob)
            # prev_observation = observation.copy()
            observation, reward, done, info = env.step(action)
            if isinstance(agent, QAgentTicTacToe):
                # if done:
                #     print(observation)
                agent.remember(prev_observation, action, observation, reward, done)
                agent.learn()

            if isinstance(opponent, QAgentTicTacToe):
                opponent.remember(prev_observation, action, observation, -reward, done)
                opponent.learn()

            if done and display:
                env.render()

            prev_observation = [observation[0].copy(), observation[1]]
            cum_reward += reward
            steps += 1

        results.append(info["result"])
        if verbose and i % 1000 == 0:
            print(f'GAME {i} winner: {info["result"]}')

    env.close()

    if save:
        for i, a in enumerate(agents):
            if isinstance(a, QAgentTicTacToe):
                save_agent(a, f'ttt_q_agent_{i+1}')

    if plot:
        plot_results(results, agents)

    return results


def plot_results(results, agents):
    # results = results[-10:]
    # np.count_nonzero(results == 1)

    n = 1000

    if len(results) <= 10000:
        p1_wins = []
        p2_wins = []
        draws = []
        for i in range(len(results)):
            if i < n:
                last_n_results = np.array(results)[:i+1]
            else:
                last_n_results = np.array(results)[i-n:i+1]

            p1_wins.append(np.count_nonzero(last_n_results == 1) / len(last_n_results) * 100)
            p2_wins.append(np.count_nonzero(last_n_results == 2) / len(last_n_results) * 100)
            draws.append(np.count_nonzero(last_n_results == 0) / len(last_n_results) * 100)
    else:
        results = np.array(results)
        p1_wins = (moving_average(results == 1, n) * 100)
        p2_wins = (moving_average(results == 2, n) * 100)
        draws = (moving_average(results == 0, n) * 100)

    # y = p1_wins
    x = range(len(p1_wins))
    # plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='r')
    # plt.scatter(x, y, marker='.')
    plt.plot(x, p1_wins, x, p2_wins, x, draws)
    plt.xlabel('game number')
    plt.ylabel(f'percent of last {n} games')
    agents_names = [str(type(a)).split("agents.")[1][:-2] for a in agents]
    title = f'{agents_names[0]} vs {agents_names[1]}'
    plt.legend([f'p1 ({agents_names[0]}) wins', f'p2 ({agents_names[1]}) wins', 'draws'])
    plt.title(title)
    # plt.plot(p2_wins, marker='.')
    # plt.plot(draws, marker='.')
    plt.grid(which='both')
    plt.show()


def save_agent(agent, filename='saved_agents/q_agent.pkl'):
    with open(filename, 'wb+') as file:
        pkl.dump(agent, file, pkl.HIGHEST_PROTOCOL)


def show(env, agents):
    play(5, env, agents, 0, display=True, plot=False)


def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# def compare_random_scales():
#     scales = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 1, 2]
#     env = gym.make('CartPole-v0')
#     games = 500
#     last_n_results_proc = 10
#     repeats = 10
#
#     for scale in scales:
#         scores = []
#         for repeat in range(repeats):
#             agent = QAgentCartPole()
#             results = play(games, env, agent, random_scale=scale, verbose=False)
#             no_of_results_to_score = int(games * last_n_results_proc / 100)
#             scores.append(np.mean(results[no_of_results_to_score:]))
#         scores_mean = np.mean(scores)
#         scores_std = np.std(scores)
#
#         print(f'random_scale: {scale} scores_mean: {scores_mean} scores_std: {scores_std}')


if __name__ == '__main__':
    register_env()
    env = gym.make('TicTacToe-v0')
    # agents = [QAgentTicTacToe(1), RandomAgentTicTacToe()]
    agents = [RandomAgentTicTacToe(), QAgentTicTacToe(2)]
    # agents[2].discount_factor = 0.9
    # agents = [RandomAgentTicTacToe(),  RandomAgentTicTacToe()]
    # agents = [QAgentTicTacToe(1), QAgentTicTacToe(2)]
    results = play(10000000, env, agents, verbose=True, plot=True, random_scale=0)
    # results = play(300, env, agents, display=False, random_scale=0, verbose=False)

    # q_table = np.array(list(agents[0].q_dict.values()))

    # print(q_table.max())
    # print(np.count_nonzero(q_table == -np.inf))
    # print(q_table.size)
    # plot_results(results[::])

    # plot_results(results[::10], agents)

    # show(env, agents)
