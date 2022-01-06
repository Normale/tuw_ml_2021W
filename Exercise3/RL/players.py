import random
import numpy as np
from typing import List


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_player_name(self):
        return self.__class__.__name__

class RandomPlayer(Player):


    def decide_action(self, state: np.array, actions: List[int]):
        return random.choice(actions)


class QLPlayer(Player):
    def __init__(self, player_id: int, q_table: np.ndarray):
        self.player_id = player_id
        self.q_table = q_table

    def decide_action(self, state: np.array, actions: List[int]):
        while True:
            action = np.argmax(self.q_table[state])
            if action in actions:
                break
            else:
                self.q_table[state + (action,)] = -10
        return action


if __name__ == '__main__':
    p1 = Player(1)
    p2 = RandomPlayer(2)
    q_table = np.load(r"Exercise3\qtables\qtable-e=100000.npy")
    p3 = QLPlayer(1, q_table)
    print(p1.get_player_name(),p2.get_player_name(),p3.get_player_name())