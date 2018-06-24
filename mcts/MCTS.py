from abc import ABC, abstractmethod
from copy import copy
import numpy as np


class TreeEdge:
    def __init__(self, prior, parent, child):
        """
        The class represents an edge between two consecutive states in a Markov Decision Process
        :param prior: prior knowledge about how good is that node
        :param parent: TreeNode, parent node
        :param child: TreeNode, child node
        """
        self.prior = prior
        self.visit_count = 0
        self.value = 0
        self.parent = parent
        self.child = child

    @abstractmethod
    def update(self, reward):
        """
        Traverse the tree up to the root so as to update value and visit count
        :param reward:
        :return:
        """
        self.value += reward
        self.visit_count += 1
        if self.parent.parent_edge is not None:
            self.parent.parent_edge.update(reward)

    def __repr__(self):
        return "Value: " + str(self.value) + \
               ", Count: " + str(self.visit_count) + \
               ", Parent: " + repr(self.parent) + \
               ", Child: " + repr(self.child)


class TreeNode(ABC):
    def __init__(self, parent_edge):
        """
        The class represents a node in Markov Decision Process
        :param parent_edge: TreeEdge or None if the node is root
        """
        self.parent_edge = parent_edge
        self.children = []
        self.edge_class = TreeEdge

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def get_available_child_nodes(self):
        """
        Get all allowed turns from the current state
        :return: iterable of (TreeNode, prior value)
        """
        return []

    @abstractmethod
    def expand(self):
        """
        Add all allowed turns to the current tree. Set prior values to child edges
        :return:
        """
        assert len(self.children) == 0, "Trying to expand node two times"

        available_children = self.get_available_child_nodes()
        for new_node, new_prior in available_children:
            new_edge = self.edge_class(new_prior, self, new_node)
            new_node.parent_edge = new_edge
            self.children.append(new_edge)

    @abstractmethod
    def pretty_print(self):
        pass

    def traverse(self):
        result = [self]
        for child in self.children:
            result.extend(child.child.traverse())
        return result

    @abstractmethod
    def get_reward(self):
        return 0

    def is_terminal(self):
        return len(self.children) == 0

    def is_finish(self):
        return len(self.get_available_child_nodes()) == 0


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    # e_x = x == x.max()
    return e_x / e_x.sum()


class MonteCarloTree:
    def __init__(self, root):
        """
        The class represents all information about a Monte-Carlo Tree, allows traversals and storage of `tactics`
        :param root: TreeNode
        """
        self.root = root

    @staticmethod
    def edge_criteria(edge, c=.3):
        """
        Selection criteria used in the first stage of MCTS
        :param edge: TreeEdge
        :param c: float, exploration coefficient
        :return: float
        """
        exploit_term = edge.value / (1 + edge.visit_count)
        parent_edge = edge.parent.parent_edge
        parent_visit_count = 0 if parent_edge is None else parent_edge.visit_count
        explore_term = c * edge.prior * (np.log(2) + np.log(1 + parent_visit_count) - np.log(1 + edge.visit_count))
        return exploit_term + explore_term

    def pretty_print(self):
        result = "Current state: \n" + self.root.pretty_print() + "\n"
        result += "Next states: \n"
        result += "\n\n".join([child.child.pretty_print() +
                               "\nValue: " + str(child.value / (1 + child.visit_count)) + "\n"
                               for child in self.root.children])
        return result

    def traverse(self):
        return self.root.traverse()

    def turn(self, node):
        """
        Take a turn
        :param node: TreeNode from where
        :return:
        """
        assert self.root == node
        action_idx = np.argmax([c.value / (1 + c.visit_count) for c in self.root.children])
        action = self.root.children[action_idx]
        action.child.children = []
        return action.child

    def fit(self, num_iter=10000):
        """
        Train a Monte-Carlo Tree Search player
        :param num_iter: number of games to play while training
        :return: None
        """
        for _ in range(num_iter):
            self._fit()

    def _fit(self):
        """
        Run a Monte-Carlo Tree Search simulation
        :return: None
        """
        # Selection phase
        cur_node = self.root
        while not cur_node.is_terminal():
            weights = softmax(np.array([self.edge_criteria(c) for c in cur_node.children]))
            action = np.random.choice(cur_node.children, p=weights)
            cur_node = action.child

        # Termination condition
        if cur_node.is_finish():
            if cur_node.parent_edge is not None:
                cur_node.parent_edge.update(cur_node.get_reward())
            return

        # Expansion
        cur_node.expand()

        # Roll-out
        sel_action = np.random.choice([c for c in cur_node.children])
        cur_node = sel_action.child

        roll_out_node = copy(cur_node)

        while not roll_out_node.is_finish():
            roll_out_node.expand()
            action = np.random.choice(roll_out_node.children)
            roll_out_node = action.child

        roll_out_node.parent_edge.update(roll_out_node.get_reward())
