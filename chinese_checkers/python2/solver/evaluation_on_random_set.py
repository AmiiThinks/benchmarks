import os
import sys
import numpy as np
import jax.numpy as jnp
import itertools
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from evaluations import evaluation
from helpers import log_helper

from mcts import mcts

logger_path = "/Users/bigyankarki/Desktop/bigyan/cc/chinese_checkers_5_5_6/python2/logs"
logger = log_helper.setup_logger('eval_search', logger_path, 'random_states.log')


solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
state = cw.CCState()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)

def remove_illegal(move_type, policy, state, cc, reverse=False):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)

    # for move in legal_moves:
    #     print(move.getFrom(), move.getTo())

    legals = pw.moves_existence(legal_moves, reverse)
    washed = legals * policy
    washed = washed / np.sum(washed)
    # washed = washed * tf.reshape((tf.ones([BOARD_SIZE, BOARD_SIZE]) - tf.eye(BOARD_SIZE)), [-1])
    # for move in legal_moves:
    #     cc.freeMove(move)

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
    
    parent_outcome = int(get_label(state, solver))
    outcome_different_from_parents = np.count_nonzero(np.array(labels) != parent_outcome)
    if outcome_different_from_parents != len(labels):
        return None, None

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
            
# function to create states from the solver
def get_random_states():
    random_states = {}

    while len(random_states) < 1000:
        random_number = np.random.randint(0, ranker.getMaxRank())
        ranker.unrank(random_number, state)

        if not cc.Done(state) and int(solver.lookup(state)) != 3:
            label, stats = process_state_labels(cc, state, solver, "MovesForward")
            if label is None:
                continue
            
            if state.getToMove():
                label = -label
                board = pw.reverse_state(state.getBoard())
            else:
                board = state.getBoard()
            
            random_states[tuple(board)] = (state.getToMove())
            print(len(random_states))
    pickle.dump(random_states, open("random_states.p", "wb"))

    return random_states


def do_evaluation(data):
    states = []
    players = []
    policy_accuracy = 0
    true_policy = []
    true_labels = []

    for state, (player, policy, label) in data.items():
        states.append(state)
        players.append(player)

        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate

        true_label, stats = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label
        # training_actions = get_action(cc, policy, cc_state, solver, "MovesForward") # get the action from the policy

        # calculate the accuracy of the policy
        # policy_accuracy += calculate_policy_accuracy(training_actions, true_optimal_actions)
        
        true_labels.append(true_label)
        true_policy.append(stats)
    
    print("Evaluating the model for {} states".format(len(states)))
    # evaluate the states in the iteration
    model_labels, model_policy = model.evaluate(states) # evaluate the state, get the label and policy from the model
    model_vs_true_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))
    
    print("Accuracy: {}".format(model_vs_true_accuracy))


if __name__ == "__main__":
    model = evaluation.Evaluation()
    random_states = get_random_states()
    # do_evaluation(random_states)
    