import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
import sys

# import file from outside directory
sys.path.append('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/')
from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
from mcts import mcts
from helpers import log_helper

# import players
from players import nn_player as nnp, random_player as rp, uct_player as uct
# from players import async_nn_player as async_nnp


# define board size
BOARD_DIM = 4
cc = cw.CCheckers()

# init logger
logger = log_helper.setup_logger('eval_players', 'eval_players.log')


class Play:
    def __init__(self, limit=True, max_turns=1000):
        self.max_turns = max_turns
        self.limit = limit

    def play(self, players):
        visited = []
        outcome = 0

        state = cw.CCState()
        cc.Reset(state)
        repeat = False

        if not self.limit:
            self.max_turns = float('inf')

        turn_count = 1

        while not cc.Done(state) and not repeat and turn_count < 101:
            player = state.getToMove()
            board = state.getBoard()

            # print("-"*100)
            # print("Current board postion: ", board)
            # print("Player turn: {}".format(player+1))

            visited.append(state.getBoard())
            current_player = players[player]
            
            
            next_action, stats = current_player.runMCTS(state, 0)
            


            cc.ApplyMove(state, next_action)
            cc.freeMove(next_action)
            # print("Player {} moved from {} to {}".format(player+ 1, next_action.getFrom(), next_action.getTo()))
            # print("-"*100)

            turn_count += 1
            repeat_count = 0

            for s in visited:
                if np.equal(np.array(state.getBoard()), np.array(s)).all():
                    repeat_count +=1
                if repeat_count > 6:
                    repeat = True

        # print(state.getBoard())
        visited.append(state.getBoard())

        # if draw
        if repeat or (not cc.Done(state) and turn_count > 100):
            outcome = 0
            # print("Draw by repetition")
        elif cc.Winner(state) == 0: # if nn_player wins. outcome = 1.
            # print("player {} won. ".format(cc.Winner(state)+1))
            outcome = 1
        elif cc.Winner(state) == 1: # if other player wins, outcome = -1
            # print("player {} won. ".format(cc.Winner(state)+1))
            outcome = -1

        return outcome


if __name__ == "__main__":
    game = Play()
    eval_mods = [0, 1, 5, 10]
    res_of_each_model = [] #[[]]

    for i in eval_mods: 
        nnpl = nnp.NNPlayer(i-1)
        uctp = uct.UCTPlayer()
        rpl = rp.RandomPlayer()

        p1 = mcts.MCTS(cc, nnpl) # nn player
        p2 = mcts.MCTS(cc, uctp) # uct player
        p3 = mcts.MCTS(cc, rpl)  # random player
        players_list = [[p1, p2], [p2, p1], [p1, p3], [p3, p1]]
        players_list_names= [['nn_{}'.format(i), 'uct'], ['uct', 'nn_{}'.format(i)], ['nn_{}'.format(i), 'rnd'], ['rnd', 'nn_{}'.format(i)]]
        outcomes = []
    
        for idx, players in enumerate(players_list):
            res = [0, 0, 0] # [p1 wins, p2 wins, draws]
            for j in range(100):
                outcome = game.play(players)
                if outcome == 1: 
                    res[0] += 1
                elif outcome == -1: 
                    res[1] += 1
                elif outcome == 0: 
                    res[2] += 1
            # print(outcomes)
            logger.info("Game: {} vs {}: {}".format(players_list_names[idx][0], players_list_names[idx][1], res))
            outcomes.append(res)
        res_of_each_model.append(outcomes)
    
    logger.info("Results of each model: {}".format(res_of_each_model))
    np.save("evaluate_results.npy", res_of_each_model)