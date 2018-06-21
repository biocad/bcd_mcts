from copy import deepcopy

import numpy as np

from tic_tac_toe.TicTacToe import TicTacToeNode
from tic_tac_toe.player import TicTacToePlayer
from mcts.MCTS import MonteCarloTree


class MCTSPlayer(TicTacToePlayer):
    def __init__(self, **kwargs):
        super().__init__()
        assert "node_turn" in kwargs, "You need to provide either `TIC` or `TAC` into the constructor"

        node_turn = kwargs["node_turn"]
        tree = kwargs.get("tree", None)
        self.size = kwargs.get("size", 3)
        if tree is None:
            root_node = TicTacToeNode(None,
                                      field=np.zeros((self.size, self.size)),
                                      node_turn=node_turn)
            self.tree = MonteCarloTree(root_node)
        else:
            self.tree = tree

    def fit(self, opponent):
        self.tree.fit(opponent)

    def turn(self, node):
        traversal = self.tree.traverse()
        node.expand()
        children_arr = [c.child for c in node.children]
        if node in traversal:
            new_root_idx = traversal.index(node)
            subtree = MonteCarloTree(deepcopy(traversal[new_root_idx]))
            turned = subtree.turn(node)
            assert turned in children_arr, "MCTS player tried to turn in a wrong way!"
            return children_arr[children_arr.index(turned)]
        else:
            print("I am going random")
            return np.random.choice(children_arr)
