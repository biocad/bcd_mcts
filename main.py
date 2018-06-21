from tic_tac_toe.player import *
from mcts.MCTS import *

SIZE = 3

if __name__ == "__main__":
    optimal_player = OptimalPlayer()
    human_player = HumanPlayer()
    monte_carlo_player = MCTSPlayer(node_turn="TIC")
    random_player = RandomPlayer()

    monte_carlo_player.fit(optimal_player)

    game_root = TicTacToeNode(None, field=np.zeros((SIZE, SIZE)), node_turn="TIC")
    cur_node = game_root

    players = [monte_carlo_player, human_player][::1]
    player_strings = ["TIC", "TAC"]
    print(cur_node.pretty_print())
    turn_i = 0

    while True:
        cur_node = players[turn_i % 2].turn(cur_node)
        print(cur_node.pretty_print())
        if cur_node.get_reward(player=player_strings[turn_i % 2]) != 0:
            print("Game over.")
            break
        turn_i += 1
