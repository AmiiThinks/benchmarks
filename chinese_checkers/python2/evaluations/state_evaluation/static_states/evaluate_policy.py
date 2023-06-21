import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import random
import multiprocessing as mp

sys.path.append("/Users/bigyankarki/Desktop/bigyan/cc/chinese_checkers_5_5_6/python2/")
from wrappers import ccwrapper as cw
from players import eval_nn_player as nnp
from wrappers import pywrapper as pw
from evaluations import evaluation
from helpers import utils
from mcts import mcts

BOARD_DIM = 5


solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
state = cw.CCState()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)


def remove_illegal(move_type, policy, state, cc, reverse=False):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)

    legals = pw.moves_existence(legal_moves, reverse)
    washed = legals * policy
    washed = washed / np.sum(washed)

    return washed, legal_moves


def get_probabilities(washed, moves, reverse=False):
    total_size = (cw.BOARD_SIZE**2)
    washed = np.reshape(washed, [total_size, total_size])
    # projection = np.argsort(pw.mapping)
    probs = []
    if reverse:
        for move in moves:
            # print(move.getFrom(), move.getTo())
            p = washed[pw.mapping[total_size - 1 - move.getFrom()], pw.mapping[total_size - 1 - move.getTo()]]
            probs.append(p)
    else:
        for move in moves:
            # print(move.getFrom(), move.getTo())
            p = washed[pw.mapping[move.getFrom()], pw.mapping[move.getTo()]]
            probs.append(p)
    return probs




def get_label(state, solver):
    result = int(solver.lookup(state))
    if result == 1:
        label = -1.0
    elif result == 2:
        label = 1.0
    else:
        label = 0.0
    return label

def process_state_labels(cc, state, solver, move_type):

    # Generate list of moves for a state and
    # find the optimal ones.
    # Create a distribution over optimal actions.

    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)
    optimal_actions = len(legal_moves) * [0]
    labels = len(legal_moves) * [0]

    for (move_ind, move) in enumerate(legal_moves):
        cc.ApplyMove(state, move)
        label = get_label(state, solver)
        cc.UndoMove(state, move)
        labels[move_ind] = label

    if len(labels) > 0:
        max_label = max(labels)
    for move_ind in range(len(labels)):
        if labels[move_ind] == max_label:
            optimal_actions[move_ind] = 1
        else:
            optimal_actions[move_ind] = 0

    stats = []
    for move_ind, move in enumerate(legal_moves):
        stats.append(mcts.Node(move.getFrom(), move.getTo(), optimal_actions[move_ind], labels[move_ind], 0))

    # for move in legal_moves:
    #     move.Print(0)
    # print(optimal_actions)

    return int(get_label(state, solver)), stats


def get_action_distribution(cc, policy, state, solver, move_type):
    legals, legal_moves = remove_illegal('MovesForward', policy, state, cc, state.getToMove())
    legals = jnp.array([legals])
    washed = jnp.where(jnp.array(legals).astype(bool), policy, -1e32*jnp.ones_like(policy))
    washed = jax.nn.softmax(washed)
    probs = np.array(get_probabilities(washed, legal_moves, state.getToMove()))
    
    return probs

def calculate_policy_accuracy(training_policy, ground_truth_policy):
    return 1 if np.sum(np.multiply(training_policy, ground_truth_policy)) > 0 else 0

def do_training_evaluation(model, nnpl, model_path, iteration, training_states):
    cc_state_list = []
    states = []
    players = []
    true_action_dist = []
    training_action_dist = []
    model_action_dist = []
    model_selection_acc = 0
    training_selection_acc = 0
    search_selection_acc = 0

    for state, (player, policy, label) in training_states.items():
        states.append(state)
        players.append(player)

        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        cc_state_list.append(cc_state)

        next_action, search_policy = nnpl.runMCTS(cc_state, 0) # run mcts on the state
        true_label, true_policy = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label

        if cc_state.getToMove():
            search_dist = pw.moves_distribution(search_policy, BOARD_DIM, reverse=True)
            sample_dist = pw.moves_distribution(true_policy, BOARD_DIM, reverse=True)
        else:
            search_dist = pw.moves_distribution(search_policy, BOARD_DIM)
            sample_dist = pw.moves_distribution(true_policy, BOARD_DIM)
            
        true_action_dist = get_action_distribution(cc, sample_dist, cc_state, solver, "MovesForward") # get the action distribution from the model
        search_action_dist = get_action_distribution(cc, search_dist, cc_state, solver, "MovesForward") # get the action distribution from the model

        model_labels, model_policy = model.evaluate(state, model_path) # evaluate the state, get the label and policy from the model

        # convert the model policy to list
        model_policy = model_policy.tolist()
        model_policy = np.array(model_policy[0])
        model_action_dist = get_action_distribution(cc, model_policy, cc_state, solver, "MovesForward") # get the action distribution from the model

        # calculate the accuracy of the model selection
        model_selection_acc += utils.calculate_accuracy(model_action_dist, true_action_dist)
        search_selection_acc += utils.calculate_accuracy(search_action_dist, true_action_dist)


    model_selection_acc = model_selection_acc / len(states)
    search_selection_acc = search_selection_acc / len(states)
    # training_selection_acc = training_selection_acc / len(states)

    return model_selection_acc, search_selection_acc

def do_evaluation(model, nnpl, model_path, iteration, training_states):
    return 0, 0
    cc_state_list = []
    states = []
    players = []
    true_action_dist = []
    model_action_dist = []
    model_selection_acc = 0
    search_selection_acc = 0

    for state, (player) in training_states.items():
        states.append(state)
        players.append(player)

        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        cc_state_list.append(cc_state)

        next_action, search_policy = nnpl.runMCTS(cc_state, 0) # run mcts on the state
        true_label, true_policy = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label

        if cc_state.getToMove():
            search_dist = pw.moves_distribution(search_policy, BOARD_DIM, reverse=True)
            sample_dist = pw.moves_distribution(true_policy, BOARD_DIM, reverse=True)
        else:
            search_dist = pw.moves_distribution(search_policy, BOARD_DIM, reverse=True)
            sample_dist = pw.moves_distribution(true_policy, BOARD_DIM)
            

        true_action_dist = get_action_distribution(cc, sample_dist, cc_state, solver, "MovesForward") # get the action distribution from the model
        search_action_dist = get_action_distribution(cc, search_dist, cc_state, solver, "MovesForward") # get the action distribution from the model

        model_labels, model_policy = model.evaluate(state, model_path) # evaluate the state, get the label and policy from the model

        # convert the model policy to list
        model_policy = model_policy.tolist()
        model_policy = np.array(model_policy[0])
        model_action_dist = get_action_distribution(cc, model_policy, cc_state, solver, "MovesForward") # get the action distribution from the model

        # calculate the accuracy of the model selection
        model_selection_acc += utils.calculate_accuracy(model_action_dist, true_action_dist)
        search_selection_acc += utils.calculate_accuracy(search_action_dist, true_action_dist)


    model_selection_acc = model_selection_acc / len(states)
    search_selection_acc = search_selection_acc / len(states)
    # training_selection_acc = training_selection_acc / len(states)

    return model_selection_acc, search_selection_acc


def make_plot(result):
    iterations = list(result.keys())

    # actions accuracy
    t_action_accuracy = [result[iteration]['t_action_accuracy'] for iteration in iterations]
    t_search_accuracy = [result[iteration]['t_search_accuracy'] for iteration in iterations]

    n_action_accuracy = [result[iteration]['n_action_accuracy'] for iteration in iterations]
    n_search_accuracy = [result[iteration]['n_search_accuracy'] for iteration in iterations]

    r_action_accuracy = [result[iteration]['r_action_accuracy'] for iteration in iterations]
    r_search_accuracy = [result[iteration]['r_search_accuracy'] for iteration in iterations]
    
    # plot action accuracy
    plt.plot(iterations, t_action_accuracy, label='Seen states wo search')
    plt.plot(iterations, n_search_accuracy, label='Seen states w search')

    plt.plot(iterations, n_action_accuracy, label='Neighboring states wo search')
    plt.plot(iterations, n_search_accuracy, label='Neighboring states w search')

    plt.plot(iterations, r_action_accuracy, label='Random states wo search')
    plt.plot(iterations, r_search_accuracy, label='Random states w search') 
    plt.xlabel('Iterations')
    plt.ylabel('Action Accuracy')
    plt.legend()
    plt.savefig('action_accuracy.png')



    

if __name__ == "__main__":
    model = evaluation.Evaluation()
    training_states = pickle.load(open("../data/sampled_training_states.p", "rb"))
    neighboring_states = pickle.load(open("../data/sampled_neighboring_states.p", "rb"))
    random_states = pickle.load(open("../data/random_states.p", "rb"))

    #sample random 10 states from each set
    n_samples = 50
    # training_states = random.sample(training_states.items(), n_samples)
    # neighboring_states = random.sample(neighboring_states.items(), n_samples)
    # random_states = random.sample(random_states.items(), n_samples)

    # take the first 10 states
    training_states = list(training_states.items())[:n_samples]
    neighboring_states = list(neighboring_states.items())[:n_samples]
    random_states = list(random_states.items())[:n_samples]

    training_states = dict(training_states)
    neighboring_states = dict(neighboring_states)
    random_states = dict(random_states)

    # models path
    models_path = '/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mods'
    models = os.listdir(models_path)
    
    iterations = [int(model.split('_')[1]) for model in models]
    iter_models = dict(zip(iterations, models))
    iterations = sorted(iterations)
    result = dict()

    for iteration in iterations:
        model_path = os.path.join(models_path, iter_models[iteration])
        nnpl = nnp.NNPlayer(model_path, iteration)
        nn_player = mcts.MCTS(cc, nnpl)

        t_action_accuracy, t_search_accuracy = do_training_evaluation(model, nn_player, model_path, iteration, training_states)
        n_action_accuracy, n_search_accuracy = do_evaluation(model, nn_player, model_path, iteration, neighboring_states)
        r_action_accuracy, r_search_accuracy = do_evaluation(model, nn_player, model_path, iteration, random_states)

        result[iteration] = {"t_action_accuracy": t_action_accuracy, "t_search_accuracy": t_search_accuracy, "n_action_accuracy": n_action_accuracy, "n_search_accuracy": n_search_accuracy, "r_action_accuracy": r_action_accuracy, "r_search_accuracy": r_search_accuracy}
        
        print("{:^25} {:^25} {:^25} {:^25} {:^25} {:^25} {:^25}".format(iteration, t_action_accuracy, t_search_accuracy, n_action_accuracy, n_search_accuracy, r_action_accuracy, r_search_accuracy))
        # print("{:^25} {:^25} {:^25}".format(iteration, t_action_accuracy, t_search_accuracy))
        pickle.dump(result, open("policy_results.p", "wb"))

    pickle.dump(result, open("policy_results.p", "wb"))
    make_plot(result)

       









   