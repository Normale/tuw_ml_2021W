import numpy as np
from .environment import Environment
from time import time, sleep

EPISODES = 40000
DISCOUNT = 0.9
LEARNING_RATE = 0.4
EPSILON = 0.5

q_table = np.zeros(shape = (3,3,3,3,3,3,3,3,3,9))

PLAYER_ID = 1
for episode in range(EPISODES):
    env = Environment(1, -1, 0)
    done = False
    while not done:
        state = env.get_state()
        while True:
            possible_actions = env.get_possible_moves()
            if np.random.random() > EPSILON:
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

    env.display()
    print(f"Episode {episode}, win?: {env.is_winner(1)}")