# self play win Jax with experience replay, and inference queue

import os, sys, json, time, warnings
import numpy as np
from multiprocessing import set_start_method, Manager, Process, Queue, Value



# import file from outside directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from players import async_nn_player as async_nnp
from games import self_play_game as spg
from helpers import log_helper, utils, iterationStats, replayBuffer
from helpers.bufferQueueManager import BufferQueueManager
from train import train
from mcts import test_mcts
from inference_engine import nn_inference

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

cc = cw.CCheckers()

# setup logger
log_path = './results/logs/'
# log_path = '/data/home/bkarki/cc/results/async_jax/logs/' # for linux
logger = log_helper.setup_logger('self_play_parallel', log_path, 'self_play_parallel.log')

# instance of playing a single game
def _play(self_play_iteration, game_count, states_q, inference_q, p_idx):
    game = spg.Play() # create game instance

    # Create players
    nnpl = async_nnp.NNPlayer(mod=self_play_iteration.value) # default player is neural network
    p1 = test_mcts.MCTS(cc, nnpl, states_q, inference_q, p_idx)
    players = [p1, p1]
    
    # calculate the total time taken for each game
    start_time = time.time()
    states, outcomes, winner = game.play(players) # play the game
    end_time = time.time()
    total_time = end_time - start_time

    return states, outcomes, total_time, winner

def play_worker(config, self_play_iteration, iteration_stats, active_play_processes,  states_q, inference_q, replay_buffer, game_count, p_idx):
    print("Play game worker {} started at {}.".format(p_idx, os.getpid())) # print the process id
    while game_count.value < config["self_play"]["iterations"]: # while the inference process is not finished
        print(game_count.value)
        states, outcomes, total_time, winner = _play(self_play_iteration, game_count, states_q, inference_q, p_idx) # play the game
        utils.add_to_buffer(states, outcomes, replay_buffer, config, self_play_iteration.value, game_count.value) # add to the buffer
        game_count.value += 1 # increment the game count
        iteration_stats.update(self_play_iteration.value, total_time, game_count.value, states, winner) # update the iteration stats
    print("Play game worker {} finished.".format(p_idx))
    active_play_processes.value -= 1
    return
            

def inference_worker(config, self_play_iteration, states_q, inference_q, active_play_processes):
    print("Inference worker started at {}".format(os.getpid()))
    states = []
    index = []

    current_self_play_iteration = 0
    nn_infer = nn_inference.NN_inference(config) # create inference instance with model 0
    nn_infer.load_model(mod=self_play_iteration.value) # load the model
    while active_play_processes.value != 0:
        # if the self play iteration has changed, then update the inference model
        if self_play_iteration.value != current_self_play_iteration: 
            current_self_play_iteration = self_play_iteration.value
            nn_infer.load_model(mod=self_play_iteration.value) # load the model

        while len(states) < active_play_processes.value:
            try:
                idx, state = states_q.get(block=True, timeout=0.1)
                states.append(state)
                index.append(idx)
            except:
                break
         
        # inference
        if len(states) > 0:
            y, policy = nn_infer.inference(states)

            for i in range(len(y)):
                inference_q[index[i]].put((y[i][0], policy[i]))

            # clear the states, index for new batch
            states = []
            index = []

    print("Inference worker exiting since play worker is finished.")
    return
    
        
def train_worker(config, replay_buffer, self_play_iteration, game_count, iteration_stats, active_play_processes):
    print("Train worker started at {}".format(os.getpid()))
    model = train.Train(config, self_play_iteration.value) # create training instance with init model 0
    while active_play_processes.value != 0:
        if replay_buffer.is_ready_for_training():
            samples = replay_buffer.sample()
            total_time = model.train(samples, self_play_iteration.value) # train the model
            self_play_iteration.value += 1 # increment the self play iteration
            game_count.value = 0 # reset the game count after each iteration
            replay_buffer.reset_new_item_counter() # reset the new item counter after each training interation to track new items added to the buffer

            # log the itearation stats
            prev_iteration = self_play_iteration.value - 1
            iteration_stats.update_training_time(prev_iteration, total_time)
            iteration_stats.log_stats(prev_iteration, logger)
    print("Train worker finished.")
    return
            

def main(config):  
    # hyperparameters
    self_play_iteration = Value('i', 0) # number of self play iterations
    game_count = Value('i', 0) # number of games played counter
    num_parallel_play_worker = config["self_play"]['num_parallel_play_worker']
    replay_buffer_size = config["self_play"]['replay_buffer_size']
    train_percentage = config["self_play"]['train_percentage']
    keep_old_data_probability = config["self_play"]['keep_old_data_probability']
    sample_size = config["self_play"]['sample_size']

    # processes states
    active_play_processes = Value('i', num_parallel_play_worker) # number of active play processes

    # register the counter with the custom manager
    BufferQueueManager.register('ReplayBuffer', replayBuffer.ReplayBuffer)
    BufferQueueManager.register('IterationStats', iterationStats.IterationStats)
    BufferQueueManager.register('Queue', Queue)
    
    with BufferQueueManager() as manager:
        # create queues
        states_q = manager.Queue() # queue for all states
        inference_q = [manager.Queue() for i in range(num_parallel_play_worker)] # queue for all inference, one for each process
        replay_buffer = manager.ReplayBuffer(capacity=replay_buffer_size, replace_old_data_probability=keep_old_data_probability, train_percentage=train_percentage, sample_size=sample_size)
        iteration_stats = manager.IterationStats(config)

        print_list = ['Iteration', 'Average Time', 'Game Count', 'Average States', 'Draw', 'Player 1 Win', 'Player 2 Win', 'Training Time']
        logger.info(" {:^12} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12}".format(*print_list))

        # create processes for playing games
        processes = []
        for p_idx in range(num_parallel_play_worker):
            p = Process(target=play_worker,  args = (config, self_play_iteration, iteration_stats, active_play_processes, states_q, inference_q[p_idx], replay_buffer, game_count, p_idx))
            processes.append(p)
        
        # create inference process
        inference_process = Process(target=inference_worker, args = (config, self_play_iteration, states_q, inference_q, active_play_processes))

        # create training process
        # train_process = Process(target=train_worker, args = (config, replay_buffer, self_play_iteration, game_count, iteration_stats, active_play_processes))

        for p in processes: # start play processes
            p.start()
        inference_process.start() # start inference process
        # train_process.start() # start training process

        for p in processes: # join play processes
            p.join()
        inference_process.join() # join inference process
        # train_process.join() # join training process

        print("Main process finished")
        return iteration_stats.get_stats()


if __name__ == "__main__":
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config', 'async_jax_config.json'))
    with open(log_path) as f:
        config = json.load(f)

    # set_start_method('spawn')

    iteration_stats = main(config)
    utils.plot_iteration_stats(config, iteration_stats)

    
