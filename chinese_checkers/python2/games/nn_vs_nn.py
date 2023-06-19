import os
import ccwrapper as cw
import pywrapper as pw
import numpy as np
import datetime
import logger as logger
import mcts
import trained_nn_player as nnp


BOARD_DIM = 4
cc = cw.CCheckers()


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

        # start the timer
        a =  datetime.datetime.now().replace(microsecond=0)
        while not cc.Done(state) and not repeat and turn_count < 101:
            player = state.getToMove()
            board = state.getBoard()

            print("-"*100)
            print("Current board postion: ", board)
            print("Player turn: {}".format(player+1))

            visited.append(state.getBoard())
            current_player = players[player]
            
            
            next_action, stats = current_player.runMCTS(state, 0)
            

            cc.ApplyMove(state, next_action)
            cc.freeMove(next_action)
            print("Player {} moved from {} to {}".format(player+ 1, next_action.getFrom(), next_action.getTo()))
            print("-"*100)

            turn_count += 1
            repeat_count = 0

            for s in visited:
                if np.equal(np.array(state.getBoard()), np.array(s)).all():
                    repeat_count +=1
                if repeat_count > 6:
                    repeat = True

        print(state.getBoard())
        visited.append(state.getBoard())

        # stop the timer
        b = datetime.datetime.now().replace(microsecond=0)
        print("Time to finish: {} s.".format(b-a))

        # if draw
        if repeat or (not cc.Done(state) and turn_count > 100):
            outcome = 0
            print("Draw by repetition")
        elif cc.Winner(state) == 0: # if nn_player wins. outcome = 1.
            print("player {} won. ".format(cc.Winner(state)+1))
            outcome = 1
        elif cc.Winner(state) == 1: # if other player wins, outcome = -1
            print("player {} won. ".format(cc.Winner(state)+1))
            outcome = -1

        return outcome


if __name__ == "__main__":
    game = Play()
    dist_eval = cw.DistEval()
    playout_module = cw.BestPlayout()
    res_of_each_model = [] #[[]]

    for i in range(30): 
        nnpl1 = nnp.NNPlayer(i)
        nnpl2 = nnp.NNPlayer(i-1)

        p1 = mcts.MCTS(cc, nnpl1)
        p2 = mcts.MCTS(cc, nnpl2)
        players_list = [[p1, p2]]
        outcomes = []
    
        for players in players_list:
            res = [0, 0, 0] # [p1 wins, p2 wins, draws]
            for j in range(100):
                outcome = game.play(players)
                if outcome == 1: 
                    res[0] += 1
                elif outcome == -1: 
                    res[1] += 1
                elif outcome == 0: 
                    res[2] += 1
            print(outcomes)
            outcomes.append(res)
        res_of_each_model.append(outcomes)

    # res_of_each_model = [[[4, 96, 0], [81, 19, 0]], [[2, 98, 0], [71, 29, 0]], [[2, 98, 0], [73, 27, 0]], [[1, 99, 0], [74, 26, 0]], [[5, 95, 0], [82, 18, 0]], [[9, 91, 0], [81, 19, 0]], [[8, 92, 0], [79, 21, 0]], [[8, 92, 0], [89, 11, 0]], [[7, 93, 0], [87, 13, 0]], [[7, 93, 0], [83, 17, 0]], [[5, 95, 0], [83, 17, 0]], [[6, 94, 0], [86, 14, 0]], [[9, 91, 0], [84, 16, 0]], [[11, 89, 0], [85, 15, 0]], [[10, 90, 0], [91, 9, 0]], [[10, 90, 0], [88, 12, 0]], [[12, 88, 0], [86, 14, 0]], [[15, 85, 0], [89, 11, 0]], [[4, 96, 0], [86, 14, 0]], [[4, 96, 0], [92, 8, 0]], [[10, 90, 0], [86, 14, 0]], [[6, 94, 0], [91, 9, 0]], [[9, 91, 0], [87, 13, 0]], [[12, 88, 0], [93, 7, 0]], [[4, 96, 0], [94, 6, 0]], [[6, 94, 0], [94, 6, 0]], [[9, 91, 0], [88, 12, 0]], [[6, 94, 0], [90, 10, 0]], [[9, 91, 0], [90, 10, 0]]]
    
    print(res_of_each_model)
    np.save("nn_vs_nn.npy", res_of_each_model)

    # log results
    # data = np.load("evaluate_results.npy", allow_pickle=True)
    
    
    # for iter, d in zip(range(30), data):
    #     print("Reuslt for nn_player(p1) vs UCT_player(p2) after {} iterations : {}".format(iter, d[0]))
    #     print("Reuslt for nn_player(p1) vs random_player(p2) after {} iterations: {}".format(iter, d[1]))
    #     print("-"*200)
