import os
import numpy as np
import sys


sys.path.append('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/')
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from mcts import mcts
from players import random_player as rp



BOARD_DIM = 4
cc = cw.CCheckers()


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
        repeat = False

        if not self.limit:
            self.max_turns = float('inf')

        turn_count = 1

        while not cc.Done(state) and not repeat and turn_count < 101:
            # print(state.getBoard(), state.getToMove() + 1)
            visited.append(state.getBoard())
            current_player = players[state.getToMove()]

            next_action, stats = current_player.runMCTS(state, 0)
            # for move in stats:
            #     print(move)

            if state.getToMove():
                # print("Player 2 move: ", next_action.getFrom(), " -> ", next_action.getTo())
                board = pw.reverse_state(state.getBoard())
                sample_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
                outcomes.append(-1)
                # value_q.put(-1)
            else:
                # print("Player 1 move: ", next_action.getFrom(), " -> ", next_action.getTo())
                board = state.getBoard()
                sample_dist = pw.moves_distribution(stats, BOARD_DIM)
                outcomes.append(1)
                # value_q.put(1)

            states.append([board, state.getToMove(), sample_dist])
            # states_q.put(board)


            cc.ApplyMove(state, next_action)
            cc.freeMove(next_action)

            turn_count += 1
            repeat_count = 0

            for s in visited:
                if np.equal(np.array(state.getBoard()), np.array(s)).all():
                    repeat_count +=1
                if repeat_count > 6:
                    repeat = True

        # print(state.getBoard())
        visited.append(state.getBoard())
        # print("Game Over!")
        # print("Turns: ", turn_count)
        # print("Winner: ", cc.Winner(state)+1)
        print("Outcomes: ", outcomes)

        if repeat or (not cc.Done(state) and turn_count > 100):
            outcomes = len(outcomes) * [0]
            print("Draw by repetition")
        elif cc.Winner(state) == 1:
            print("Player 2 wins")
            outcomes = [-1 * x for x in outcomes]

        print("Outcomes: ", outcomes)


        # print("Outcomes: ", outcomes)
        return states, outcomes, cc.Winner(state)+1

if __name__ == "__main__":
    # play a game
    game = Play()

     # Create the players
    rpl = rp.RandomPlayer()
    random_player = mcts.MCTS(cc, rpl)  # random player
    players = [random_player, random_player]
    
    s, out, winner = game.play(players)