from abc import ABC, abstractmethod


class TicTacToePlayer(ABC):
    def __init__(self):
        """
        This class encapsulates an instance of something able to play TicTacToe
        """
        pass

    @abstractmethod
    def turn(self, node):
        """
        Put a tic or tac into a given field and return the result
        :param: TicTacNode
        :return: TicTacNode
        """
        pass
