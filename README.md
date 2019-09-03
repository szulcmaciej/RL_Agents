# RL_Agents
Learning and experimenting with reinforcement learning algorithms using OpenAI gym environments

# TicTacToe environment
## Environment
Custom OpenAI env was created for Tic Tac Toe game.
Implementation was inspired by [CartPole-v0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py).
Environment includes GUI rendering of the game.

## Results
Two agents were implemented:
- RandomAgentTicTacToe - an agent choosing random legal action.
- QAgentTicTacToe - an agent using q-learning to learn actions quality

As there are many states that are impossible to reach (or almost imposiible if you let agents do illegal actions), Q-learning was implemented using a dict to store q-values for state-actions. A game state is only added to dict if occured during agent's game.

### 3x3 board results
![QAgent vs Random](https://i.imgur.com/5BYJHbu.png "QAgent vs Random")
![Random vs QAgent](https://i.imgur.com/ux8Urf6.png "Random vs QAgent")
![Random vs Random](https://i.imgur.com/66G8fYA.png "Random vs Random")
![QAgent vs QAgent](https://i.imgur.com/Si9pTGo.png "QAgent vs QAgent")
---
### 4x4 board results
#### Random vs QAgent
#### 300 000 games
![Random vs QAgent](https://raw.githubusercontent.com/szulcmaciej/RL_Agents/master/results/Random_vs_QAgent_4x4.png "Random vs QAgent")
#### 1 000 000 games
![Random vs QAgent](https://raw.githubusercontent.com/szulcmaciej/RL_Agents/master/results/Random_vs_QAgent_4x4_million_games.png "Random vs QAgent")
#### 10 000 000 games
![Random vs QAgent](https://raw.githubusercontent.com/szulcmaciej/RL_Agents/master/results/Random_vs_QAgent_4x4_10_million_games.png "Random vs QAgent")
---
# CartPole-v0 environment
![CartPole-v0](https://cdn-images-1.medium.com/max/1600/1*oMSg2_mKguAGKy1C64UFlw.gif "CartPole-v0")

First version of q-learning algorithm was implemented using gym's CartPole-v0 environment. The biggest challenge was finding a proper discrete representation for continuous features of game state.

