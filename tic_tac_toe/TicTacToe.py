from copy import copy
import numpy as np

from mcts.MCTS import TreeNode, TreeEdge

turn_seq_dict = {"TIC": "TAC", "TAC": "TIC"}  # Sequence of moves
tic_num_dict = {"TIC": 1, "TAC": 2, "FREE": 0}  # Signs of players in the field
TIC = tic_num_dict["TIC"]
TAC = tic_num_dict["TAC"]
FREE = tic_num_dict["FREE"]
tic_char_dict = {TIC: "x", TAC: "o", FREE: " "}  # Signs of players used in pretty printing


class TicTacToeEdge(TreeEdge):
    def update(self, reward):
        self.value += reward
        self.visit_count += 1
        if self.parent.parent_edge is not None:
            self.parent.parent_edge.update(-reward)


class TicTacToeNode(TreeNode):
    """
    Implementation of Monte-Carlo Tree Node related to Tic-Tac Toe game.
    """
    def __init__(self, parent_edge, **kwargs):
        super().__init__(parent_edge)
        assert "field" in kwargs, "You must pass a tic-tac toe field into the constructor"
        assert "node_turn" in kwargs, "You must pass `TIC` or `TAC` as a `node_turn` into constructor"
        assert kwargs["node_turn"] in ["TIC", "TAC"], "Node turn could be either `TIC` or `TAC`"

        self.field = kwargs["field"]
        assert self.field.shape[0] == self.field.shape[0], "Field must have square shape"
        self.field_size = self.field.shape[0]
        self.node_turn = kwargs["node_turn"]
        self.edge_class = TicTacToeEdge

    def get_available_child_nodes(self):
        result = []

        if self.victory("TIC") or self.victory("TAC"):
            return result

        empty_pair = np.where(self.field == FREE)
        for r, c in zip(*empty_pair):
            new_field = copy(self.field)
            new_field[r, c] = TIC if self.node_turn == "TIC" else TAC
            next_turn = turn_seq_dict[self.node_turn]
            new_node = TicTacToeNode(None, field=new_field, node_turn=next_turn)
            result.append((new_node, 1))
        return result

    def __eq__(self, other):
        return (self.field == other.field).all()

    def pretty_print(self):
        row_strings = []
        for row in self.field:
            row_strings.append(" | ".join([tic_char_dict[el] for el in row]))
        return "\n{}\n".format("---" * self.field.shape[0]).join(row_strings)

    def expand(self):
        super().expand()

    def is_filled(self):
        return (self.field == 0).sum() == 0

    def victory(self, player):
        assert player in ["TIC", "TAC"]
        rows, columns = np.where(self.field == tic_num_dict[player])
        row_victory = any([(rows == i).sum() == self.field_size for i in range(self.field_size)])
        col_victory = any([(columns == i).sum() == self.field_size for i in range(self.field_size)])
        main_diagonal = np.sum([r == c for r, c in zip(rows, columns)]) == self.field_size
        side_diagonal = np.sum([(r + c) == (self.field_size - 1) for r, c in zip(rows, columns)]) == self.field_size
        return any([row_victory, col_victory, main_diagonal, side_diagonal])

    def defeat(self, player):
        return self.victory(turn_seq_dict[player])

    def get_reward(self):
        """
        The last turn in a game could either be 'victory' or 'tie'
        You could never make a turn, immediately leading you to defeat
        :return: 1 if 'victory' 0 otherwise
        """
        return 1 if self.defeat(self.node_turn) else 0
