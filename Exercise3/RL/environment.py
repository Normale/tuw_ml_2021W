import numpy as np
from typing import Tuple, List


class Environment:
    def __init__(self, win_reward, lose_reward, draw_reward, enemy, env_first: bool = True):
        self.board = np.zeros(9, dtype=np.uint8)
        self.enemy = enemy
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        if env_first:
            actions = self.get_possible_moves()
            enemy_action = self.enemy.decide_action(self.get_state(), actions)
            self.try_move(self.enemy.player_id, enemy_action)
    def get_state(self):
        return tuple(self.board)

    def get_possible_moves(self):
        self.display
        return np.where(self.board == 0)[0]

    def _check_diagonal(self):
        indices = [[0,4,8], [2,4,6]]
        for i in indices:
            unique = np.unique(self.board[i])
            if unique.size == 1 and unique[0] != 0:
                return unique[0]
        return None
        

    def _check_rows(self):
        indices = [[0,1,2], [3,4,5], [6,7,8]]
        for i in indices:
            unique = np.unique(self.board[i])
            if unique.size == 1 and unique[0] != 0:
                return unique[0]
        return None

    def _check_columns(self):
        indices = [[0,3,6], [1,4,7], [2,5,8]]
        for i in indices:
            unique = np.unique(self.board[i])
            if unique.size == 1 and unique[0] != 0:
                return unique[0]
        return None

    def is_winner(self, player_id):
        checklist = [self._check_columns(), self._check_diagonal(), self._check_rows()]
        for check in checklist:
            if check is not None: 
                return check == player_id

    def is_move_possible(self):
        # It actually counts number of zeroes!
        zero_count = np.count_nonzero(self.board==0)
        return zero_count != 0

    def try_move(self, player_id, action):
        if action not in self.get_possible_moves():
            print(f"AGENT {player_id} DID FORBIDDEN MOVE: {action}"
                    f"moves: {self.get_possible_moves()}")
            return False
        self.board[action] = player_id
        return True

    def display(self):
        print('----')
        for x in np.split(self.board, 3):
            print(x)    

    def step(self, player_id, action) -> Tuple[np.array, int, bool]:
        '''
        returns [state, reward, done]
        '''
        if not self.try_move(player_id, action):
            ULTRA_PUNISHMENT_LMAO = -100
            return [None, ULTRA_PUNISHMENT_LMAO, True]

        if self.is_winner(player_id):
            return self.get_state(), self.win_reward, True
        
        if not self.is_move_possible():
            return self.get_state(), self.draw_reward, True

        actions = self.get_possible_moves()
        enemy_action = self.enemy.decide_action(self.get_state(), actions)
        self.try_move(self.enemy.player_id, enemy_action)

        if self.is_winner(self.enemy.player_id):
            return self.get_state(), self.lose_reward, True

        if not self.is_move_possible():
            return self.get_state(), self.draw_reward, True

        return self.get_state(), self.draw_reward, False