import os
import sys
# import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import random
import haiku as hk


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from models import AZeroModel
from players import eval_nn_player as nnp
from helpers import utils
from evaluations import evaluation

from mcts import mcts


solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
state = cw.CCState()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)
BOARD_DIM = 5
BOARD_SIZE = BOARD_DIM**2


def permute_states(board_slice_length, num_pieces):
    substate_strings = []
    substates = itertools.permutations(num_pieces * '2' + (board_slice_length - num_pieces)*'0')
    for substate in substates:
        if ''.join(substate) not in substate_strings:
            substate_strings.append(''.join(substate))
    return substate_strings


def remove_illegal(move_type, policy, state, cc, reverse=False):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)

    # for move in legal_moves:
    #     print(move.getFrom(), move.getTo())

    legals = pw.moves_existence(legal_moves, reverse)
    washed = jnp.array(legals) * policy
    washed = washed / jnp.sum(washed)
    # washed = washed * tf.reshape((tf.ones([BOARD_SIZE, BOARD_SIZE]) - tf.eye(BOARD_SIZE)), [-1])
    # for move in legal_moves:
    #     cc.freeMove(move)

    return washed, legal_moves

def get_probabilities(washed, moves, reverse=False):
    total_size = (cw.BOARD_SIZE**2)
    washed = jnp.reshape(washed, [total_size, total_size])
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

    return int(get_label(state, solver)), stats, optimal_actions


def rollout_terminal_states(cc, state, solver, move_type):
    pass


def get_action(cc, policy, state, solver, move_type):
    legals, legal_moves = remove_illegal('MovesForward', policy, state, cc, state.getToMove())
    legals = jnp.array([legals])
    washed = jnp.where(jnp.array(legals).astype(bool), policy, -1e32*jnp.ones_like(policy))
    washed = jax.nn.softmax(washed)
    probs = jnp.array(get_probabilities(washed, legal_moves, state.getToMove()))
    
    # select move with highest probability
    move_ind = jnp.argwhere(probs == jnp.amax(probs)).flatten()

    # convert index of probs to 1 for those in move_ind
    probs = jnp.zeros_like(probs)
    probs = probs.at[move_ind].set(1.0)
    # probs[move_ind] = 1.0

    return probs

def calculate_policy_accuracy(training_policy, ground_truth_policy):
    # print(training_policy)
    # print(ground_truth_policy)
    # print(type(training_policy))
    # print(type(ground_truth_policy))

    ground_truth_policy = jnp.array(ground_truth_policy)

    # return 1 if index if index value is 1 for both policies
    return 1 if jnp.sum(jnp.multiply(training_policy, ground_truth_policy)) > 0 else 0




# load the training states dictionary
def load_dict_of_states():
    # states =  pickle.load(open('unique_training_states.p', 'rb'))
    states =  pickle.load(open('iteration_training_states.p', 'rb'))
    return states

def load_result_data():
    data =  pickle.load(open('result.pkl', 'rb'))
    return data

def make_accuracy_graph():
    result = load_result_data()
    iterations = list(result.keys())
    accuracy = [x["accuracy"] for x in result.values()]
    training_label_accuracy = [x["training_label_accuracy"] for x in result.values()]
    ground_truth_label_accuracy = [x["ground_truth_label_accuracy"] for x in result.values()]
    training_policy_accuracy = [x["training_policy_accuracy"] for x in result.values()]
    model_policy_accuracy = [x["model_policy_accuracy"] for x in result.values()]

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0, 0].plot(iterations, accuracy, label="Training vs ground truth")
    ax[0, 0].plot(iterations, training_label_accuracy, label="Model vs training labels")
    ax[0, 0].plot(iterations, ground_truth_label_accuracy, label="Model vs ground truth")
    ax[0, 0].set_title("Label accuracy comparison")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].legend()

    ax[0, 1].plot(iterations, training_policy_accuracy, label="Training policy accuracy")
    ax[0, 1].plot(iterations, model_policy_accuracy, label="Model policy accuracy")
    ax[0, 1].set_title("Policy accuracy comparison")
    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()


    plt.savefig("training_accuracy.png")
    plt.close()
    return


def do_evaluation(iteration, data, result, nnp):
    policy_accuracy = 0
    print("Iteration: {}, Data size: {}".format(iteration, len(data)))

    for state, (player, policy, label) in data:
        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate

        # Run MCTS from the current state to get the search policy
        next_action, stats = nnp.runMCTS(cc_state, 0)

        if cc_state.getToMove():
            sample_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
        else:
            sample_dist = pw.moves_distribution(stats, BOARD_DIM)

        mcts_actions = get_action(cc, sample_dist, cc_state, solver, "MovesForward") # get the action from the policy
        # model_action = get_action(cc, policy, cc_state, solver, "MovesForward")
        # # # get the ground truth policy
        _, _, optimal_actions = process_state_labels(cc, cc_state, solver, 'MovesForward')

        
        policy_accuracy += calculate_policy_accuracy(mcts_actions, optimal_actions)

        result[str(iteration)] = { "accuracy": policy_accuracy / len(data) }          

if __name__ == "__main__":
    model = evaluation.Evaluation()
    nnpl = nnp.NNPlayer("/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mod_166", 166)
    nn_player = mcts.MCTS(cc, nnpl)
    
    trainings_states = load_dict_of_states()
    result = {}

    global_data = {}
    for i in range(len(trainings_states)):
        global_data.update(trainings_states[str(i)])
        if i % 10 == 0:
            # get random 100000 states
            random_data = random.sample(global_data.items(), min(len(global_data), 1000))
            do_evaluation(i, random_data, result=result, nnp=nn_player)
            print(result)
            # global_data = {}

    # sort result by iteration
    # result = dict(sorted(result.items()))
    # print(result)

    # with open('result.pkl', 'wb') as fp:
    #     pickle.dump(result, fp)
    
    # make_accuracy_graph()
            