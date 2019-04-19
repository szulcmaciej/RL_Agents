from q_learning_tictactoe import register_env, RandomAgentTicTacToe, QAgentTicTacToe
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


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
        if verbose:
            print(f'GAME {i} winner: {info["result"]}')

    env.close()

    if save:
        for i, a in enumerate(agents):
            if isinstance(a, QAgentTicTacToe):
                save_agent(a, f'ttt_q_agent_{i+1}')

    if plot:
        plot_results(results, agents)

    return results



def main():
    register_env()
    env = gym.make('TicTacToe-v0')
    rec = VideoRecorder(env, 'video.mp4')
    agents = [RandomAgentTicTacToe(), RandomAgentTicTacToe()]
    play(1, rec.env, agents, 0, display=True)


if __name__ == '__main__':
    main()
