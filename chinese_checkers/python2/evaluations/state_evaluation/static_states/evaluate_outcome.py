import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import random
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import math

BOARD_DIM = 5

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# print(os.getcwd())
# print(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append("/Users/bigyankarki/Desktop/bigyan/cc/chinese_checkers_5_5_6/python2/")
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from evaluations import evaluation
from mcts import mcts
from helpers import log_helper, utils

solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
solver = cw.Solver(solved_data, True, False)

logger_path = "/Users/bigyankarki/Desktop/bigyan/cc/chinese_checkers_5_5_6/python2/logs"
logger = log_helper.setup_logger('eval_outcome', logger_path, 'eval_outcome.log')


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

    return int(get_label(state, solver)), stats


def get_action_distribution(cc, policy, state):
    legals, legal_moves = remove_illegal('MovesForward', policy, state, cc, state.getToMove())
    legals = jnp.array([legals])
    washed = jnp.where(jnp.array(legals).astype(bool), policy, -1e32*jnp.ones_like(policy))
    washed = jax.nn.softmax(washed)
    probs = np.array(get_probabilities(washed, legal_moves, state.getToMove()))
    # probs = probs / np.sum(probs)
    return probs

def calculate_accuracy(model_dist, true_dist):
    model_action = np.argmax(model_dist)
    max_true_action = max(true_dist)
    acc = int(true_dist[model_action] == max_true_action)
    return acc

def do_training_evaluation(model, model_path, iteration, iter_states, name):
    states = []
    cc_states = []
    players = []
    true_policy = []
    true_labels = []

    train_policies = []
    train_labels = []

    model_policy_accuracy = 0
    train_policy_accuracy = 0

    for state, (player, train_policy, train_label) in iter_states.items():
        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        cc_states.append(cc_state)

        true_label, stats = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label

        if cc_state.getToMove():
            sample_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
        else:
            sample_dist = pw.moves_distribution(stats, BOARD_DIM)

        # calculate the accuracy of the policy
        states.append(state)
        players.append(player)
        true_policy.append(sample_dist)
        true_labels.append(true_label)

        train_policies.append(train_policy)
        train_labels.append(train_label)

    model_labels, model_policy = model.evaluate(states, model_path) # evaluate the state, get the label and policy from the model
    model_vs_true_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))
    train_vs_true_accuracy = jnp.mean(jnp.where(jnp.array(train_labels) == jnp.array(true_labels), 1, 0))

    # calculate the accuracy of the model policy
    for state, player, m_policy, train_policy, t_policy in zip(cc_states, players, model_policy, train_policies, true_policy):
        true_action_dist = get_action_distribution(cc, t_policy, state) # get the action distribution from the model
        train_action_dist = get_action_distribution(cc, train_policy, state) # get the action distribution from the model

        train_policy_accuracy += calculate_accuracy(train_action_dist, true_action_dist)

         # convert the model policy to list
        m_policy = model_policy.tolist()
        m_policy = np.array(model_policy[0])
        model_action_dist = get_action_distribution(cc, m_policy, state) # get the action distribution from the model

        model_policy_accuracy += calculate_accuracy(model_action_dist, true_action_dist)

    model_policy_accuracy = model_policy_accuracy / len(states)
    train_policy_accuracy = train_policy_accuracy / len(states)
    return model_vs_true_accuracy,  model_policy_accuracy, train_vs_true_accuracy, train_policy_accuracy


def do_evaluation(model, model_path, iteration, iter_states, name):
    states = []
    cc_states = []
    players = []
    true_policy = []
    true_labels = []
    model_policy_accuracy = 0

    for state, (player) in iter_states.items():
        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        cc_states.append(cc_state)

        true_label, stats = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label

        if cc_state.getToMove():
            sample_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
        else:
            sample_dist = pw.moves_distribution(stats, BOARD_DIM)

        # calculate the accuracy of the policy
        states.append(state)
        players.append(player)
        true_policy.append(sample_dist)
        true_labels.append(true_label)


    model_labels, model_policy = model.evaluate(states, model_path) # evaluate the state, get the label and policy from the model
    model_vs_true_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))

    # calculate the accuracy of the model policy
    for state, player, m_policy, t_policy in zip(cc_states, players, model_policy, true_policy):
        true_action_dist = get_action_distribution(cc, t_policy, state) # get the action distribution from the model

        # convert the model policy to list
        m_policy = model_policy.tolist()
        m_policy = np.array(model_policy[0])
        model_action_dist = get_action_distribution(cc, m_policy, state) # get the action distribution from the model

        model_policy_accuracy += calculate_accuracy(model_action_dist, true_action_dist)

    model_policy_accuracy = model_policy_accuracy / len(states)
    return model_vs_true_accuracy, model_policy_accuracy


def make_plot(result):
    iterations = list(result.keys())

    # label accuracy
    t_label_accuracy = [result[iteration]['t_label_accuracy'] for iteration in iterations]
    t_train_label_accuracy = [result[iteration]['t_train_label_accuracy'] for iteration in iterations]
    n_label_accuracy = [result[iteration]['n_label_accuracy'] for iteration in iterations]
    r_label_accuracy = [result[iteration]['r_label_accuracy'] for iteration in iterations]
 
    # actions accuracy
    t_action_accuracy = [result[iteration]['t_action_accuracy'] for iteration in iterations]
    n_action_accuracy = [result[iteration]['n_action_accuracy'] for iteration in iterations]
    r_action_accuracy = [result[iteration]['r_action_accuracy'] for iteration in iterations]
    
    # plot label accuracy
    plt.plot(iterations, t_label_accuracy, label='Seen States Accuracy')
    plt.plot(iterations, t_train_label_accuracy, label='Training Data Accuracy')
    plt.plot(iterations, n_label_accuracy, label='Neighboring States Accuracy')
    plt.plot(iterations, r_label_accuracy, label='Random States Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Label Accuracy')
    plt.legend()
    plt.savefig('../../../../results/test_label1.png')
    plt.clf()

    # plot action accuracy
    plt.plot(iterations, t_action_accuracy, label='Training Accuracy')
    plt.plot(iterations, n_action_accuracy, label='Neighboring Accuracy')
    plt.plot(iterations, r_action_accuracy, label='Random Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Action Accuracy')
    plt.legend()
    plt.savefig('../../../../results/test_policy1.png')


if __name__ == "__main__":
    model = evaluation.Evaluation()
    training_states = pickle.load(open("../data/sampled_training_states.p", "rb"))
    neighboring_states = pickle.load(open("../data/sampled_neighboring_states.p", "rb"))
    random_states = pickle.load(open("../data/random_states.p", "rb"))

    #sample random 10 states from each set
    n_samples = len(training_states)
    training_states = random.sample(training_states.items(), n_samples)
    neighboring_states = random.sample(neighboring_states.items(), n_samples)
    random_states = random.sample(random_states.items(), n_samples)

    training_states = dict(training_states)
    neighboring_states = dict(neighboring_states)
    random_states = dict(random_states)

    # models path
    models_path = '/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mods'
    models = [f for f in os.listdir(models_path) if f.startswith('mod')]

    
    iterations = [int(model.split('_')[1]) for model in models]
    iter_models = dict(zip(iterations, models))
    iterations = sorted(iterations)
    result = dict()

    print_res = ["Iteration", "Training Label Accuracy", "Training Train Label Accuracy", "Neighboring Label Accuracy", "Random Label Accuracy", "Training Action Accuracy", "Training Train Action Accuracy", "Neighboring Action Accuracy", "Random Action Accuracy"]
    logger.info('{:^25} {:^25} {:^25} {:^25} {:^25} {:^25} {:^25} {:^25} {:^25}'.format(*print_res))
    for iteration in iterations:
        model_path = os.path.join(models_path, iter_models[iteration])

        t_label_accuracy, t_action_accuracy, t_train_label_accuracy, t_train_action_accuracy = do_training_evaluation(model, model_path, iteration, training_states, "training states")
        n_label_accuracy, n_action_accuracy = do_evaluation(model, model_path, iteration, neighboring_states, "neighboring states")
        r_label_accuracy, r_action_accuracy = do_evaluation(model, model_path, iteration, random_states, "random states")

        result[iteration] = {'t_label_accuracy': t_label_accuracy, 't_train_label_accuracy': t_train_label_accuracy, 'n_label_accuracy': n_label_accuracy, 'r_label_accuracy': r_label_accuracy, 't_action_accuracy': t_action_accuracy, 't_train_action_accuracy': t_train_action_accuracy, 'n_action_accuracy': n_action_accuracy, 'r_action_accuracy': r_action_accuracy}
        logger.info('{:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f}'.format(iteration, t_label_accuracy, t_train_label_accuracy, n_label_accuracy, r_label_accuracy, t_action_accuracy, t_train_action_accuracy, n_action_accuracy, r_action_accuracy))
        pickle.dump(result, open("eval_results_on_ground_truth.p", "wb"))

    result = pickle.load(open("eval_results_on_ground_truth.p", "rb"))

    make_plot(result)










   