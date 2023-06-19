import os
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
import numpy as np
import logging

BOARD_DIM = 5
cc = cw.CCheckers()

# get the logger
logger = logging.getLogger('self_play')

class Play:
    def __init__(self, limit=True, max_turns=1000):
        self.max_turns = max_turns
        self.limit = limit

    def play(self, players):

        states = []
        visited = []
        outcomes = []

        state = cw.CCState()
        cc.Reset(state)
        value = 0.0
        repeat = False

        if not self.limit:
            self.max_turns = float('inf')

        turn_count = 1


        while not cc.Done(state) and not repeat and turn_count < 101:
            visited.append(state.getBoard())
            current_player = players[state.getToMove()]

            next_action, stats = current_player.runMCTS(state, 0)

            if state.getToMove():
                board = pw.reverse_state(state.getBoard())
                sample_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
                outcomes.append(-1)
            else:
                board = state.getBoard()
                sample_dist = pw.moves_distribution(stats, BOARD_DIM)
                outcomes.append(1)

            states.append([board, state.getToMove(), sample_dist])

            cc.ApplyMove(state, next_action)
            cc.freeMove(next_action)

            turn_count += 1
            repeat_count = 0

            for s in visited:
                if np.equal(np.array(state.getBoard()), np.array(s)).all():
                    repeat_count +=1
                if repeat_count > 6:
                    repeat = True

        visited.append(state.getBoard())
       
        if repeat or (not cc.Done(state) and turn_count > 100):
            outcomes = len(outcomes) * [0]
            print("Draw by repetition")
        elif cc.Winner(state) == 1:
            outcomes = [-1 * x for x in outcomes]

        return states, outcomes, cc.Winner(state)+1