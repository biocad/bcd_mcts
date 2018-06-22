import numpy as np

from tic_tac_toe.player import TicTacToePlayer
from mcts.MCTS import softmax


class OptimalPlayer(TicTacToePlayer):
    tic_symbol = "X"
    tac_symbol = "O"

    def __init__(self):
        super().__init__()
        self.policy_dict = dict()

        with open("data/policy.dat", "r") as f:
            _ = f.readline()
            for line in f:
                self.policy_dict[line.split("\t")[0]] = float(line.split("\t")[1])

    def turn(self, node):
        node_values = []
        assert len(node.children) == 0
        cur_turn = node.node_turn
        cur_sign = 1 if cur_turn == "TIC" else -1

        node.expand()

        for child in node.children:
            child = child.child
            cur_char = self.tic_symbol if cur_turn == "TIC" else self.tac_symbol
            new_state = str((tuple(child.field.astype(int).flatten().tolist()), cur_char))
            new_value = self.policy_dict[new_state]
            node_values.append((child, new_value))
        # return max(node_values, key=lambda t: cur_sign * t[1])[0]

        probabilities = softmax([cur_sign * t[1] for t in node_values])
        # p = (p == p.max()).astype(int)
        # print(p)
        p = np.zeros((len(node_values)))
        p[np.argmax(probabilities)] = 1
        return np.random.choice([n[0] for n in node_values], p=p)
