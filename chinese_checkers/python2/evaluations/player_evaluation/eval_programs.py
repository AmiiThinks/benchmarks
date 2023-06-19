# program to run evaluations one very hour

import numpy as np
import os
import sys
import argparse
from multiprocessing import Process

# import file from outside directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
from mcts import mcts
from helpers import log_helper

# import players
from players import jax_nn_player as nnp, random_player as rp, uct_player as uct


# define board size
BOARD_DIM = 4
cc = cw.CCheckers()

# init logger
logger_path = "/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/evaluations/"
logger = log_helper.setup_logger('eval_players', logger_path, 'eval_players.log')


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

            visited.append(state.getBoard())
            current_player = players[player]
            
            next_action, stats = current_player.runMCTS(state, 0)
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

# nn vs random player and uct player
def play_regular_games(config, nn_instance):
    game_play_instance = Play()

    # define uctp player and random player
    nnpl = nnp.NNPlayer(config, nn_instance+1)
    uctp = uct.UCTPlayer()
    rpl = rp.RandomPlayer()
    nn_player = mcts.MCTS(cc, nnpl)
    uct_player = mcts.MCTS(cc, uctp) # uct player
    random_player = mcts.MCTS(cc, rpl)  # random player

    # define games to play
    games = [[nn_player, uct_player], [uct_player, nn_player], [nn_player, random_player], [random_player, nn_player]]
    games_names = ['nn_{}_vs_uct'.format(nn_instance), 'uct_vs_nn_{}'.format(nn_instance), 'nn_{}_vs_random'.format(nn_instance), 'random_vs_nn_{}'.format(nn_instance)]

    for idx, game in enumerate(games):
        res = [0, 0, 0] # [p1 wins, p2 wins, draws]
        for j in range(2):
            outcome = game_play_instance.play(game)
            if outcome == 1: 
                res[0] += 1
            elif outcome == -1: 
                res[1] += 1
            elif outcome == 0: 
                res[2] += 1
        logger.info("Game: {}: {}".format(games_names[idx], res))


# nn_vs_nn game play worker
def play_nn_vs_nn(config, nn_instance):
    game_play_instance = Play()

    # define uctp player and random player
    nnpl1 = nnp.NNPlayer(config[0], nn_instance[0]+1) # sigle nn_player_bot
    nnpl2 = nnp.NNPlayer(config[1], nn_instance[1]+1) # parallel nn_player_bot
    nn_player1 = mcts.MCTS(cc, nnpl1)
    nn_player2 = mcts.MCTS(cc, nnpl2)

    # define games to play
    games = [[nn_player1, nn_player2], [nn_player2, nn_player1]]
    games_names = ['single_nn_bot_{} vs parallel_nn_bot_{}'.format(nn_instance[0], nn_instance[1]), 'parallel_nn_bot_{} vs single_nn_bot_{}'.format(nn_instance[1], nn_instance[0])]

    for idx, game in enumerate(games):
        res = [0, 0, 0] # [p1 wins, p2 wins, draws]
        for j in range(1):
            outcome = game_play_instance.play(game)
            if outcome == 1: 
                res[0] += 1
            elif outcome == -1: 
                res[1] += 1
            elif outcome == 0: 
                res[2] += 1
        logger.info("Game: {} : {}".format(games_names[idx], res))



def main(config, nn_instances):
    regular_games_processes = [] # define process for regular games
    for i in range(len(nn_instances)):
        p = Process(target=play_regular_games, args=(config[i], nn_instances[i], ))
        regular_games_processes.append(p)
        p.start()

    nn_vs_nn_games_processes = [] # define processes for single_play_nn vs parallel_play_nn
    for i in range(len(nn_instances)-1):
        p = Process(target=play_nn_vs_nn, args=(config, nn_instances))
        nn_vs_nn_games_processes.append(p)
        p.start()

    # join processes
    for p in regular_games_processes:
        p.join()

    for p in nn_vs_nn_games_processes:
        p.join()

    return 0

if __name__ == '__main__':
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--counter', type=int, default=0, help='No. of hours passed since the start of the experiment')
    args = parser.parse_args()


    # get the latest nn_player from both single_play and parallel_play
    single_play_trained_params_path = "/Users/bigyankarki/Desktop/bigyan/results/jax/single/trained_models/"
    parallel_play_trained_path = "/Users/bigyankarki/Desktop/bigyan/results/async_jax/parallel/trained_models/"

    # find the latest mod and opt files
    try:
        single_play_latest_model_file = max([single_play_trained_params_path + f for f in os.listdir(single_play_trained_params_path) if f.startswith('mod')], key=os.path.getctime)
        parallel_play_latest_model_file = max([parallel_play_trained_path + f for f in os.listdir(parallel_play_trained_path) if f.startswith('mod')], key=os.path.getctime)
    except ValueError:
        logger.info("No trained models found. Exiting...")
        exit()

    # get the model number from the file name
    single_play_latest_model_number = int(single_play_latest_model_file.split('_')[-1].split('.')[0])
    parallel_play_latest_model_number = int(parallel_play_latest_model_file.split('_')[-1].split('.')[0])

    nn_instances = [single_play_latest_model_number, parallel_play_latest_model_number]
    nn_instance_file_path = []
    for path in [single_play_trained_params_path, parallel_play_trained_path]:
        config = {
            "file":{
                "trained_model_params_path": path
            }
        }
        nn_instance_file_path.append(config)
    
    # main(nn_instance_file_path, nn_instances)

   

    
