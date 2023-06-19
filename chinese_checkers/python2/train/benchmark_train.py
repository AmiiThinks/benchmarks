import os
import json
import time
import sys

# import file from outside directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from wrappers import ccwrapper as cw
from players import jax_nn_player as nnp
from games import self_play_game as spg
from helpers import log_helper, utils, replayBuffer, iterationStats
from mcts import mcts
import train


cc = cw.CCheckers()

# Load the hyperparameters from the file
log_path = './results/logs/'
logger = log_helper.setup_logger('self_play', log_path, 'single_self_play.log')



def _train(model, data, self_play_iteration):
    total_training_time = model.train(data, self_play_iteration) # train the model
    return total_training_time


def main(config):
    self_play_iteration = 0

    for i in range(20):
        model = train.Train(config, self_play_iteration) # create training instance with init model 0
        data = utils.create_dataset_from_folder(config['file']['training_data_path']) # create dataset from folder

        # number of self play games to play
        total_training_time = _train(model, data, self_play_iteration)  # Train the model
        print(i, total_training_time)
    return "Training Benchmark Complete."


if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config', 'async_jax_config.json'))
    with open(config_path) as f:
        config = json.load(f)
        logger.info(config)

    main(config)

    
        



        













        
        
