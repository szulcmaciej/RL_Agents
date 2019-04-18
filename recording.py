from q_learning_tictactoe import play, register_env, RandomAgentTicTacToe
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def main():
    register_env()
    env = gym.make('TicTacToe-v0')
    rec = VideoRecorder(env, 'video.mp4')
    agents = [RandomAgentTicTacToe(), RandomAgentTicTacToe()]
    play(1, rec, agents, 0)


if __name__ == '__main__':
    main()
