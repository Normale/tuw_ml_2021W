import numpy as np
from .environment import Environment
from .players import RandomPlayer, QLPlayer
import matplotlib.pyplot as plt
from pathlib import Path


filepath = r"Exercise3\qtables\qtable-e=100000.npy"
q_table = np.load(filepath)

EPISODES = 10000


rand_player = RandomPlayer(2)
ql_player = QLPlayer(1, q_table)

draw_win_lose = [0, 0, 0]
win_percent = []
draw_percent = []
lose_percent = []

for episode in range(EPISODES):
    env = Environment(1, 0, -1, rand_player)
    done = False
    while not done:
        state = env.get_state()
        possible_actions = env.get_possible_moves()
        action = ql_player.decide_action(state, possible_actions)
        new_state, result, done = env.step(ql_player.player_id, action)

    if result < -1:
        print(f"did incorrect move from state {state}")
        result = -1

    draw_win_lose[result] += 1
    print(f"win percent: {draw_win_lose}")
    win_percent.append(draw_win_lose[1] / (episode+1))
    lose_percent.append(draw_win_lose[-1] / (episode+1))
    draw_percent.append(draw_win_lose[0] / (episode+1))


x = range(EPISODES)
plt.plot(x, win_percent, c='g')
plt.plot(x, lose_percent, c='r')
plt.plot(x, draw_percent, c='y')

graph_dir_name = "Graphs"
save_path = Path.cwd() / 'Exercise3' / graph_dir_name
Path.mkdir(save_path, exist_ok=True)
plt.savefig(f"{save_path}/ql_results_e={EPISODES}.png")
