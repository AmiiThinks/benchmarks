import ccwrapper as cw
from players import nn_player as nnp, random_player as rp, human_player as hp, uct_player as up
import self_play_game as spg
from mcts import mcts
from helpers import log_helper

import parser
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Play a game of Chinese Checkers')
parser.add_argument('--p1', type=str, default='up', help='Player 1 type (human, random, nn, up)')
parser.add_argument('--p2', type=str, default='nnp', help='Player 2 type (human, random, nn, up)')
parser.add_argument('--limit', type=bool, default=True, help='Limit the number of turns')
parser.add_argument('--max_turns', type=int, default=1000, help='Maximum number of turns')
args = parser.parse_args()

BOARD_DIM = 4
cc = cw.CCheckers()

class Play:
    def __init__(self, limit=True, max_turns=1000):
        self.max_turns = max_turns
        self.limit = limit

    def play(self, players):
        state = cw.CCState()
        cc.Reset(state)
        repeat = False

        if not self.limit:
            self.max_turns = float('inf')

        turn_count = 1

        while not cc.Done(state) and not repeat and turn_count < 101:
            player = state.getToMove()
            board = state.getBoard()

            print("-"*100)
            print("Current board postion: ", board)
            print("Player turn: {}, {}".format(player+1, players[player]._name))
            
            p1 = mcts.MCTS(cc, players[0])
            p2 = players[1]

            # if agent in nn_player
            if player == 0:
                next_action, stats = p1.runMCTS(state, 0)
            else:
                next_action = p2.next_move(cc, state)
            
            print("Player {} moved from {} to {}".format(player+ 1, next_action.getFrom(), next_action.getTo()))
            print("-"*100)
            cc.ApplyMove(state, next_action)
            cc.freeMove(next_action)
        
        print("player {} won. ".format(cc.Winner(state)+1))


        return



if __name__ == "__main__":
    logger = log_helper.setup_logger('game_{}_vs_{}'.format(args.p1, args.p2), 'game_{}_vs_{}.log'.format(args.p1, args.p2))

    game = Play(limit=args.limit, max_turns=args.max_turns)


    
    # number of self play games to play
    for i in range(1): # 30-50
        logger.info('Starting iteration: ' + str(i))

    
    game.play(players)

