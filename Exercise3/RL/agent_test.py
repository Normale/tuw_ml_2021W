from tkinter.tix import ExFileSelectBox
import numpy as np
from .environment import Environment
from .players import ManualPlayer, RandomPlayer, QLPlayer
import matplotlib.pyplot as plt
from pathlib import Path


filepath = r"Exercise3\qtables\qtable.npy"

EPISODES = 1000


rand_player = RandomPlayer(2)
ql_player = QLPlayer(1, np.load(filepath))
manual = ManualPlayer(2)

draw_win_lose = [0, 0, 0]
win_percent = []
draw_percent = []
lose_percent = []

for episode in range(EPISODES):
    env = Environment(1, -1, 0, manual, env_first=True)
    done = False
    while not done:
        state = env.get_state()
        possible_actions = env.get_possible_moves()
        action = ql_player.decide_action(state, possible_actions)
        new_state, result, done = env.step(ql_player.player_id, action)
    
    if result < -1:
        print(f"did incorrect move from state {state}")
        result = -1

    for x in np.split(np.array(state), 3):
        print(x)
    if result == 1:
        print("Agent won")
    elif result == 0:
        print("Game ended in a draw")
    elif result == -1:
        print("Agent lost")
    draw_win_lose[result] += 1
    print(f"win percent: {draw_win_lose}")
    win_percent.append(draw_win_lose[1] / (episode+1))
    lose_percent.append(draw_win_lose[-1] / (episode+1))
    draw_percent.append(draw_win_lose[0] / (episode+1))
    print(f"win: {100*draw_win_lose[1] / (episode+1):.2f}%, lose: {100*draw_win_lose[-1] / (episode+1):.2f}%, "
    f"draw: {100*draw_win_lose[0] / (episode+1):.2f}%")


x = range(EPISODES)
plt.title("Agent performance against random player")
plt.ylabel('%')
plt.xlabel('episodes')
plt.legend(['win', 'lose', 'draw'])
plt.plot(x, win_percent, c='g')
plt.plot(x, lose_percent, c='r')
plt.plot(x, draw_percent, c='y')

graph_dir_name = "Graphs"
save_path = Path.cwd() / 'Exercise3' / graph_dir_name
Path.mkdir(save_path, exist_ok=True)
plt.savefig(f"{save_path}/ql_results_e={EPISODES}.png")
