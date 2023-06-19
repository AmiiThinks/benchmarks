import os
import numpy as np
import json
import time
import sys

# import file from outside directory
# sys.path.append('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/')

from wrappers import ccwrapper as cw
from players import nn_player as nnp
from games import self_play_game as spg
from helpers import train, log_helper, utils
from mcts import mcts


cc = cw.CCheckers()

if __name__ == "__main__":
    # Load the hyperparameters from the file
    with open('./config/config.json') as f:
        config = json.load(f)
        logger = log_helper.setup_logger('self_play', 'self_play.log')
        logger.info('Loaded with config: ')
        logger.info(config)


    game = spg.Play()

    # number of self play games to play
    for i in range(config["self_play"]["iterations"]): # 30-50
        logger.info('Starting iteration: ' + str(i))
        
        # Create the players
        nnpl = nnp.NNPlayer(mod=i) # default player is neural network
        p1 = mcts.MCTS(cc, nnpl)
        p2 = mcts.MCTS(cc, nnpl) 
        players = [p1, p2]
        
        # calculate the total time taken for each game
        total_time = 0

        # Play the no. of games specified in the hyperparams self_play.games
        logger.info('{:^7} {:^7} {:^8}'.format('Game', 'Winner', 'Time'))
        for j in range(config["self_play"]["games"]): #1000
            # logger.info('Starting game: ' + str(j))
            start_time = time.time()

            states, outcomes, winner = game.play(players) # play the game

            # Save the states and outcomes
            utils.create_ndrecord(states, outcomes, config["file"]["training_data"]+'iteration_{}'.format(i), '/game_{}.npy'.format(j))

            end_time = time.time()
            time_taken = end_time - start_time
            total_time += time_taken
            # logger.info('Game: ' + str(j) + ' took: ' + str(time_taken) + ' seconds')
            logger.info('{:^7} {:^7} {:^8}'.format(str(j), winner, str(time_taken)))
        
        # average time taken for each game
        avg_time = total_time / config["self_play"]["games"]
        logger.info('Average time taken for each game: ' + str(avg_time))

        # Train the neural network
        train.train(i)
