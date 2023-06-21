# program to run evaluations one very hour

import numpy as np
import os
import sys
from multiprocessing import Process
import time

# import file from outside directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
from mcts import mcts, uct_mcts
from helpers import log_helper
from eval_accuracy import Evaluation

# import players
from players import eval_nn_player as nnp, random_player as rp, uct_player as uct


# define board size
BOARD_DIM = 5
cc = cw.CCheckers()

# init logger
logger_path = "/Users/bigyankarki/Desktop/bigyan/cc/chinese_checkers_5_5_6/python2/evaluations/"
logger = log_helper.setup_logger('eval_players', logger_path, 'eval_players.log')


class Play:
    def __init__(self, limit=True, max_turns=1000):
        self.max_turns = max_turns
        self.limit = limit

    def play(self, players):
        visited = []
        outcome = 0
        start_time = time.time()

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

        play_time = time.time() - start_time

        return outcome, play_time

# nn vs random player and uct player
def play_regular_games(hour, model_path, model_number):
    game_play_instance = Play()
    model_name = model_path.split('/')[-1]
    model_name = "single play model" if model_name.startswith("single") else "parallel play model"
    print("Playing games for {} {} at hour {}, pid: {}.".format(model_name, model_number, hour, os.getpid()))


    # define uctp player and random player
    nnpl = nnp.NNPlayer(model_path, model_number)
    uctp = uct.UCTPlayer()
    rpl = rp.RandomPlayer()
    nn_player = mcts.MCTS(cc, nnpl)
    uct_player = uct_mcts.MCTS(cc, uctp) # uct player
    # random_player = mcts.MCTS(cc, rpl)  # random player

    # define games to play
    # games = [[nn_player, uct_player], [uct_player, nn_player], [nn_player, random_player], [random_player, nn_player]]
    games = [[nn_player, uct_player], [uct_player, nn_player]]
    # games_names = ['nn_{}_vs_uct'.format(model_number), 'uct_vs_nn_{}'.format(model_number), 'nn_{}_vs_random'.format(model_number), 'random_vs_nn_{}'.format(model_number)]
    games_names = ['nn_{}_vs_uct'.format(model_number), 'uct_vs_nn_{}'.format(model_number)]

    for idx, game in enumerate(games):
        res = [0, 0, 0] # [p1 wins, p2 wins, draws]
        for j in range(100):
            outcome, play_time = game_play_instance.play(game)
            print("Game: {}, outcome: {}, Time: {}".format(j, outcome, play_time))
            if outcome == 1: 
                res[0] += 1
            elif outcome == -1: 
                res[1] += 1
            elif outcome == 0: 
                res[2] += 1
        logger.info("For {} At hour {} Game: {}: {}".format(model_name, hour, games_names[idx], res))

    # clear memory
    del nn_player
    del uct_player
    # del random_player
    del nnpl
    del uctp
    # del rpl

    del game_play_instance

    return


# nn_vs_nn game play worker
def play_nn_vs_nn(hour, model_paths, model_numbers):
    print("Comapring model {} and model {} at hour {} PID: {}".format(model_numbers[0], model_numbers[1], hour, os.getpid()))
    game_play_instance = Play()

    # define uctp player and random player
    nnpl1 = nnp.NNPlayer(model_paths[0], model_numbers[0]) # sigle nn_player_bot
    nnpl2 = nnp.NNPlayer(model_paths[1], model_numbers[1]) # parallel nn_player_bot
    nn_player1 = mcts.MCTS(cc, nnpl1)
    nn_player2 = mcts.MCTS(cc, nnpl2)

    # define games to play
    games = [[nn_player1, nn_player2], [nn_player2, nn_player1]]
    games_names = ['single_nn_bot_{} vs parallel_nn_bot_{}'.format(model_numbers[0], model_numbers[1]), 'parallel_nn_bot_{} vs single_nn_bot_{}'.format(model_numbers[1], model_numbers[0])]

    for idx, game in enumerate(games):
        res = [0, 0, 0] # [p1 wins, p2 wins, draws]
        for j in range(100):
            outcome = game_play_instance.play(game)
            if outcome == 1: 
                res[0] += 1
            elif outcome == -1: 
                res[1] += 1
            elif outcome == 0: 
                res[2] += 1
        logger.info("At hour {} Game: {} : {}".format(hour, games_names[idx], res))
    return 

# evaluate accuracy of the models
def eval_accuracy(hours, model_paths, model_numbers):
    evaluation = Evaluation()
    result = []
    for i, (a, b, c) in enumerate(zip(hours, model_paths, model_numbers)):
        for j, (x, y) in enumerate(zip(b, c)):
            result.append([a, x, y])
    
    for hour, model_path, model_number in result:
        model_name = model_path.split("/")[-1]
        model_name = model_name.startswith("single_play") and "single_play" or "parallel_play"
        print("Evaluating {} model {} at hour {} PID:{}.".format(model_name, model_number, hour, os.getpid()))
        accuracy, value_loss, policy_loss = evaluation.evaluate(model_path)
        logger.info({"hour": hour, "model_name": model_name, "model_number": model_number, "accuracy": accuracy, "value_loss": value_loss, "policy_loss": policy_loss})
    
    return
    



def main(hours, model_paths, model_numbers):
    result = []
    for i, (a, b, c) in enumerate(zip(hours, model_paths, model_numbers)):
        for j, (x, y) in enumerate(zip(b, c)):
            result.append([a, x, y])

    highest_model_path = None
    highest_model_number = None
    
    # find the highest model number and its path for parallel play
    for hour, model_path, model_number in result:
        if "parallel_play" in model_path:
            if highest_model_number is None or highest_model_number < model_number:
                highest_model_number = model_number
                highest_model_path = model_path

    parallel_play_models = []
    for hour, model_path, model_number in result:
        if "parallel_play" in model_path:
            parallel_play_models.append([hour, [model_path, highest_model_path], [model_number, highest_model_number]])
    
    
    # regular_games_processes = [] # define process for regular games
    # for hour, model_path, model_number in result:
    #     p = Process(target=play_regular_games, args=(hour, model_path, model_number, ))
    #     regular_games_processes.append(p)
    #     p.start()

    # nn_vs_nn_games_processes = [] # define processes for single_play_nn vs parallel_play_nn
    # for i in range(len(comparisons_model_number)):
    #     p = Process(target=play_nn_vs_nn, args=(hours[i], model_paths[i], model_numbers[i]))
    #     nn_vs_nn_games_processes.append(p)
    #     p.start()

    # final_nn_vs_other_parallel_nn_games_processes = [] # define processes for single_play_nn vs parallel_play_nn
    # for i in range(len(parallel_play_models)):
    #     p = Process(target=play_nn_vs_nn, args=(parallel_play_models[i][0], parallel_play_models[i][1], parallel_play_models[i][2]))
    #     final_nn_vs_other_parallel_nn_games_processes.append(p)
    #     p.start()

    eval_process = Process(target=eval_accuracy, args=(hours, model_paths, model_numbers))
    eval_process.start()

    # join processes
    # for p in regular_games_processes:
    #     p.join()

    # for p in nn_vs_nn_games_processes:
    #     p.join()

    # for p in final_nn_vs_other_parallel_nn_games_processes:
    #     p.join()

    eval_process.join()

    return 0

if __name__ == '__main__':
    # get the latest nn_player from both single_play and parallel_play
    latest_models_path = "/Users/bigyankarki/Desktop/bigyan/results/latest_models/"
    hours = []
    comparisons_path = []
    comparisons_model_number = []

    # loop through the folders in latest_models_path
    for folder in os.listdir(latest_models_path):
        if folder.startswith('.'):
            continue
        hour_number = int(folder.split("_")[-1])
        # loop thorugh the folder name starting with hour in the folder
        if hour_number%10 == 0:
            # get inside the folder
            hour_folder_path = os.path.join(latest_models_path, folder)
            try:
                # grab file that starts with single_play and parallel_play
                single_play_model = max([os.path.join(hour_folder_path, f) for f in os.listdir(hour_folder_path) if f.startswith('single_play')], key=os.path.getctime)
                parallel_play_model = max([os.path.join(hour_folder_path, f) for f in os.listdir(hour_folder_path) if f.startswith('parallel_play')], key=os.path.getctime)
                hour = int(folder.split('_')[-1])
                hours.append(hour)
                comparisons_path.append([single_play_model, parallel_play_model])
                # get the model number from the file name
                single_play_latest_model_number = int(single_play_model.split('_')[-1].split('.')[0])
                parallel_play_latest_model_number = int(parallel_play_model.split('_')[-1].split('.')[0])
                comparisons_model_number.append([single_play_latest_model_number, parallel_play_latest_model_number])
            except ValueError:
                logger.info("No trained models found. Exiting...")


    
    main(hours, comparisons_path, comparisons_model_number)

   

    
