import numpy as np
from .environment import Environment, RandomPlayer
from time import time, sleep
from pathlib import Path
import matplotlib.pyplot as plt

PLAYER_ID = 1
EPISODES = 100000
DISCOUNT = 0.9
LEARNING_RATE = 0.4
EPSILON = 0.5
DISPLAY = False
q_table = np.zeros(shape = (3,3,3,3,3,3,3,3,3,9))

draw_win_lose = [0,0,0]
win_percent = []
draw_percent = []
lose_percent = []

for episode in range(EPISODES):
    env = Environment(1, -1, 0, RandomPlayer(player_id=2))
    done = False
    while not done:
        state = env.get_state()
        while True:
            possible_actions = env.get_possible_moves()
            rand = np.random.random()
            if rand > EPSILON:
                action = np.argmax(q_table[state])
            else:
                action = np.random.randint(0,9)
            if action not in possible_actions:
                q_table[state + (action,)] = -10
            else:
                break
        new_state, reward, done = env.step(PLAYER_ID, action)
        current_q = q_table[state + (action,)]
        if not done:
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state + (action,)] = new_q
        else:
            q_table[state + (action,)] = reward

    draw_win_lose[reward] += 1
    win_percent.append(draw_win_lose[1] / (episode+1))
    lose_percent.append(draw_win_lose[-1] / (episode+1))
    draw_percent.append(draw_win_lose[0] / (episode+1))
    if DISPLAY:
        env.display()
    print(f"Episode {episode}, rand: {rand <= EPSILON} win?: {env.is_winner(1)}")




x = range(EPISODES)
plt.plot(x, win_percent, c='g')
plt.plot(x, lose_percent, c='r')
plt.plot(x, draw_percent, c='y')

graph_dir_name = "Graphs"
save_path = Path.cwd() / 'Exercise3' / graph_dir_name
Path.mkdir(save_path, exist_ok=True)
plt.savefig(f"{save_path}/q_learning_e={EPISODES}.png")

qtables = "qtables"
qtables_path = Path.cwd() / 'Exercise3' / qtables
Path.mkdir(qtables_path, exist_ok=True)
np.save(f"{qtables_path}/qtable-e={EPISODES}.npy", q_table)
