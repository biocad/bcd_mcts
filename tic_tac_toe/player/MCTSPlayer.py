import numpy as np

from tic_tac_toe.TicTacToe import TicTacToeNode
from tic_tac_toe.player import TicTacToePlayer
from mcts.MCTS import MonteCarloTree


class MCTSPlayer(TicTacToePlayer):
    def __init__(self, **kwargs):
        super().__init__()
        assert "node_turn" in kwargs, "You need to provide either `TIC` or `TAC` into the constructor"

        node_turn = kwargs["node_turn"]
        root_path = kwargs.get("root", None)
        self.size = kwargs.get("size", 3)
        if root_path is None:
            root_node = TicTacToeNode(None,
                                      field=np.zeros((self.size, self.size)),
                                      node_turn=node_turn)
            self.tree = MonteCarloTree(root_node)
        else:
            self.tree = MonteCarloTree(root_path)

    def fit(self):
        self.tree.fit()

    def turn(self, node):
        traversal = self.tree.traverse()
        node.expand()
        children_arr = [c.child for c in node.children]
        if node in traversal:
            new_root_idx = traversal.index(node)
            weights = [self.tree.edge_criteria(c, 0) for c in traversal[new_root_idx].children]
            if len(weights) == 0:
                # We don't know what to do.
                return np.random.choice(children_arr)
            sel_node = traversal[new_root_idx].children[np.argmax(weights)].child
            assert sel_node in children_arr

            return children_arr[children_arr.index(sel_node)]
        else:
            print("I am going random")
            return np.random.choice(children_arr)
