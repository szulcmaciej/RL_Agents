from envs.tic_tac_toe import TicTacToeEnv
from gym.envs.registration import registry, register, spec
import gym

env = TicTacToeEnv()

register(
    id='TicTacToe-v0',
    entry_point='envs.tic_tac_toe:TicTacToeEnv',
    max_episode_steps=200,
    reward_threshold=1.0,
)

env = gym.make('TicTacToe-v0')
observation = env.reset()

print(observation)


def play(games, random_scale=0.4, verbose=True):

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

            # random_prob = random_scale * ((games - i) / games) ** 4
            action = int(input('Action: '))
            observation, reward, done, info = env.step(action)
            # reward = reward if not done else -10
            # agent.remember(prev_observation, action, observation, reward, done)
            # agent.experience_replay()
            # prev_observation = observation

            cum_reward += reward
            steps += 1

        results.append(steps)
        # print(f'game {i} finished after {steps} steps' + '*'.rjust(steps, '|'))

    env.close()

    # plot_results(results)

    # print(f'ones: {sum(actions)}/{len(actions)}')

    return results

play(1)