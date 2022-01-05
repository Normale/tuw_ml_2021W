import random
import numpy as np
from typing import List

class RandomPlayer:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def decide_action(self, state: np.array, actions: List[int]):
        return random.choice(actions)


class QLPlayer:
    def __init__(self, player_id: int, q_table: np.array):
        self.player_id = player_id
        self.q_table = q_table

    def decide_action(self, state: np.array, actions: List[int]):
        return np.argmax(self.q_table[state])
