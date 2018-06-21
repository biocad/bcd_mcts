import numpy as np

from tic_tac_toe.player import TicTacToePlayer


class RandomPlayer(TicTacToePlayer):
    tic_symbol = "X"
    tac_symbol = "O"

    def turn(self, node):
        assert len(node.children) == 0
        node.expand()
        return np.random.choice([c.child for c in node.children])
