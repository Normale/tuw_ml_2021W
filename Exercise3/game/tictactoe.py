import random


class Board:

    def __init__(self):
        self.state = {}  # Init state (empty dictionary)
        for y in range(3):
            for x in range(3):
                self.state[(x, y)] = Player.neutral

    def __str__(self):
        keys = list(self.state.keys())

        string = "\nBOARD:\n"
        rows = []
        for i in range(3):
            sliced = keys[3*i:3*i+3]
            rows.append(' | '.join(['{}'.format(self.state[key]) for key in sliced]))
        string += "\n---------\n".join(rows)
        return string

    def place(self, symbol, x, y):
        if not self.is_valid_action(x, y):
            return False

        self.state[x, y] = symbol
        return True

    def is_valid_action(self, x, y):
        if self.state[x, y] == Player.neutral:
            return True
        return False

    def is_winner(self, player):
        symbol = player.get_symbol()
        state = self.state

        return (any(all(state[i, j] == symbol for j in range(3)) for i in range(3)) or
                any(all(state[i, j] == symbol for i in range(3)) for j in range(3)) or
                all(state[i, i] == symbol for i in range(3)) or
                all(state[i, 2 - i] == symbol for i in range(3)))

    def is_draw(self):
        state = self.state

        return all(state[i, j] != Player.neutral for i in range(3) for j in range(3))


class Player:
    neutral = " "

    def __init__(self, game, symbol):
        self.symbol = symbol
        self.game = game

    def __str__(self):
        return "Player {}".format(self.symbol)

    def get_symbol(self):
        return self.symbol

    def place(self, coord):
        self.game.place(self, coord)

    @staticmethod
    def get_random_coord():
        return random.randint(0, 2), random.randint(0, 2)

    def get_random_coord_allowed(self):
        return random.choice(self.game.get_allowed_moves())

    def place_random_allowed(self):
        self.place(self.get_random_coord_allowed())


class Game:

    def __init__(self, symbol_p1, symbol_p2):
        self.board = Board()
        self.player1 = Player(self, symbol_p1)
        self.player2 = Player(self, symbol_p2)

    def __str__(self):
        string = "\n===================\n   PRINTING GAME\n===================\nPLAYERS:\n"
        for player in self.get_players():
            string += str(player) + "\n"
        string += str(self.board)
        if self.is_finished():
            string += "\n\nSTATUS:\n" + self.print_status()

        return string

    def place(self, player, coord):
        self.board.place(player.get_symbol(), coord[0], coord[1])

    def get_players(self):
        return self.player1, self.player2

    def get_allowed_moves(self):
        return list(self.board.state.keys())

    def is_finished(self):
        finished = False
        for player in self.get_players():
            finished = self.board.is_winner(player) or finished
        return finished or self.board.is_draw()

    def print_status(self):
        if not self.is_finished():
            return "in progress"
        if self.board.is_draw():
            return "draw"
        for player in self.get_players():
            if self.board.is_winner(player):
                return str(player) + " won"

    def get_result(self):
        if self.board.is_draw():
            return Player.neutral
        for player in self.get_players():
            if self.board.is_winner(player):
                return player.symbol

    def sim_random(self, printing=True):
        while not self.is_finished():
            for player in self.get_players():
                player.place_random_allowed()
        if printing is True:
            print("\n\nGAME FINISHED\n\n")
            print(self)

