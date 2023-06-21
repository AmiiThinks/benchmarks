import sys
import random as rnd
import numpy as np


# import file from outside directory
sys.path.append('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/')
from wrappers import pywrapper as pw


class RandomPlayer:
    def __init__(self):
        self._name = "random"

    @staticmethod
    def get_next_action(cc, state):
        forward_moves = cc.getMovesForward(state)
        forward_moves = pw.ll_to_list(forward_moves)
        next_action = rnd.choice(forward_moves)
        next_action.setNextMove(None)

        # for m in forward_moves:
        #     print(m.getFrom(), m.getTo())

        # for action in forward_moves:
        #     if not action == next_action:
        #         cc.freeMove(action)

        return next_action

    def evaluate_state(self, state, cc, depth=0):
        if cc.Done(state):
            v_s = -1
            return v_s
        if depth == 10:
            return 0
        else:
            action = self.get_next_action(cc, state)
            _from = action.getFrom()
            to = action.getTo()
            # print(_from, to, state.getBoard())
            cc.ApplyMove(state, action)
            outcome = self.evaluate_state(state, cc, depth+1)
            cc.UndoMove(state, action)
            # print(_from, to, state.getBoard())
            cc.freeMove(action)
            return -outcome

    def expand_actions(self, cc, state, depth):
        forward_moves = cc.getMovesForward(state)
        forward_moves = pw.ll_to_list(forward_moves)
        # for action in forward_moves:
        #     cc.freeMove(action)

        return forward_moves, len(forward_moves) * [0]

    def select_action(self, node, depth):
        return rnd.choice(node.children)

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name
