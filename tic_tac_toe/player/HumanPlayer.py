import numpy as np

from .TicTacToePlayer import *


class HumanPlayer(TicTacToePlayer):
    """
    This class encapsulates input-output related to taking turns by human
    """
    def turn(self, node):
        r, c = [int(i) for i in input().split()]
        turns = [t[0] for t in node.get_available_child_nodes()]
        diff_fields = [turn for turn in turns if np.where(turn.field != node.field) == (r, c)]
        assert len(diff_fields) == 1, "Illegal move!"
        return diff_fields[0]
