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
from train import train


cc = cw.CCheckers()

# Load the hyperparameters from the file
log_path = '/Users/bigyankarki/Desktop/bigyan/results/single_jax/logs/'
# log_path = '/data/home/bkarki/cc/results/single_jax/logs/' # for linux
logger = log_helper.setup_logger('self_play', log_path, 'single_self_play.log')


# instance of playing a single game
def _play(config, nnpl, replay_buffer, iteration_stats, game, self_play_iteration, game_count):

    p1 = mcts.MCTS(cc, nnpl)
    players = [p1, p1]
    
    # calculate the total time taken for each game
    start_time = time.time()
    states, outcomes, winner = game.play(players) # play the game
    total_time = time.time() - start_time
    utils.add_to_buffer(states, outcomes, replay_buffer, config, self_play_iteration, game_count) # add the game to the replay buffer
    iteration_stats.update(self_play_iteration, total_time, game_count+1, states, winner) # update the iteration stats

    return states, outcomes, total_time, winner


def _train(model, replay_buffer, self_play_iteration, iteration_stats):
    replay_buffer.reset_new_item_counter() # reset the new item counter after each training interation to track new items added to the buffer
    samples = replay_buffer.sample() # sample from the replay buffer
    total_training_time = model.train(samples, self_play_iteration) # train the model

    # log the itearation stats
    iteration_stats.update_training_time(self_play_iteration, total_training_time)
    iteration_stats.log_stats(self_play_iteration, logger)
    return "Training complete."


def main(config):
    replay_buffer_size = config["self_play"]['replay_buffer_size']
    train_percentage = config["self_play"]['train_percentage']
    keep_old_data_probability = config["self_play"]['keep_old_data_probability']
    sample_size = config["self_play"]['sample_size']

    self_play_iteration = 0
    game_count = 0

    replay_buffer = replayBuffer.ReplayBuffer(capacity=replay_buffer_size, replace_old_data_probability=keep_old_data_probability, train_percentage=train_percentage, sample_size=sample_size) # create replay buffer instance
    iteration_stats = iterationStats.IterationStats(config)
    model = train.Train(config, self_play_iteration) # create training instance with init model 0
    game = spg.Play() # create game
    nnpl = nnp.NNPlayer(config) # default player is neural network
    nnpl.load_model(self_play_iteration) # load the model

    

    print_list = ['Iteration', 'Average Time', 'Game Count', 'Average States', 'Draw', 'Player 1 Win', 'Player 2 Win', 'Training Time']
    logger.info(" {:^12} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12}".format(*print_list))

    # number of self play games to play
    while self_play_iteration < config["self_play"]['iterations']:
        _ = _play(config, nnpl, replay_buffer, iteration_stats, game, self_play_iteration, game_count) # play the game
        game_count += 1 # increment the game count
        if replay_buffer.is_ready_for_training():
            _ = _train(model, replay_buffer, self_play_iteration, iteration_stats)  # Train the model
            self_play_iteration += 1 # increment the self play iteration
            game_count = 0 # reset the game count after each self play iteration  
            nnpl.load_model(self_play_iteration) # load the new model after each iteration
            
    
    # plot graphs after self_play is complete
    iteration_stats = iteration_stats.get_stats()
    utils.plot_iteration_stats(config, iteration_stats)
    return "Single self play with buffer queue finished."


if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config', 'jax_single_play_config.json'))
    with open(config_path) as f:
        config = json.load(f)
        logger.info(config)

    main(config)

    
        



        













        
        
