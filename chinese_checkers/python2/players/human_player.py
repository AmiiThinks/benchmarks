import sys
from turtle import forward
sys.path.append('../')
import pywrapper as pw
import ccwrapper as cw
import random as rnd
import numpy as np


class MePlayer:
    def __init__(self):
        self._name = "me_player"

    def next_move(self, cc, state):
        forward_moves = cc.getMovesForward(state)
        forward_moves = pw.ll_to_list(forward_moves)

        for idx, moves in enumerate(forward_moves):
            print("Select {} to get from {} to {}".format(idx, forward_moves[idx].getFrom(), forward_moves[idx].getTo()))
        
        # take input from user from command line
        player_action = input("Select action: ")
        next_action = forward_moves[int(player_action)]

        return next_action

# if __name__ == "__main__":
#     cc = cw.CCheckers()
#     state = cw.CCState()
#     cc.Reset(state)

#     rp = MePlayer()
#     print(state.getBoard())
