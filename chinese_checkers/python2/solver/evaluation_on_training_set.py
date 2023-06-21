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
from helpers import utils
from evaluations import evaluation

from mcts import mcts


solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
state = cw.CCState()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)


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
    training_label_accuracy = [
        x["training_label_accuracy"] for x in result.values()
    ]
    ground_truth_label_accuracy = [
        x["ground_truth_label_accuracy"] for x in result.values()
    ]
    training_policy_accuracy = [
        x["training_policy_accuracy"] for x in result.values()
    ]
    model_policy_accuracy = [x["model_policy_accuracy"] for x in result.values()]

    # Creating the first subplot for label accuracies
    plt.subplot(2, 1, 1)
    plt.plot(iterations, accuracy, label='Training comp. ground truth')
    plt.plot(iterations, training_label_accuracy, label='Model comp. training labels')
    plt.plot(iterations, ground_truth_label_accuracy, label='Model comp. ground truth')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Label Accuracies')
    plt.legend()


    # Creating the second subplot for policy accuracy
    plt.subplot(2, 1, 2)
    plt.plot(iterations, training_policy_accuracy, label='Training Policy Accuracy')
    plt.plot(iterations, model_policy_accuracy, label='Model Policy Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Policy Accuracy')
    plt.legend()

    # Adjusting the spacing between subplots
    plt.tight_layout()

    plt.savefig("training_accuracy.png")
    plt.close()
    return


def do_evaluation(iteration, data, result):
    print("Evaluating on iteration: ", i)


    states = []
    players = []
    training_policy = []
    training_labels = []
    training_policy_accuracy = 0
    model_policy_accuracy = 0
    true_policy = []
    true_labels = []
    true_optimal_actions_arr = []

    for state, (player, policy, label) in data:
        states.append(state)
        players.append(player)
        training_policy.append(policy)
        training_labels.append(label)

        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        
        policy = jnp.array(policy)
        true_label, stats, true_optimal_actions = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label
        training_actions = get_action(cc, policy, cc_state, solver, "MovesForward") # get the action from the policy

        true_optimal_actions_arr.append(true_optimal_actions) # append the true optimal actions

        # calculate the accuracy of the policy
        training_policy_accuracy += calculate_policy_accuracy(training_actions, true_optimal_actions)
        
        true_labels.append(true_label)
        true_policy.append(stats)

    # evaluate the states in the iteration
    print("Evaluating the model on iteration {}, number of states: {}".format(iteration, len(data))) 
    model_labels, model_policy = model.evaluate(states) # evaluate the state, get the label and policy from the model
    print("Evaluation done")
    model_vs_training_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(training_labels), 1, 0))
    model_vs_true_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))
    training_vs_true_accuracy = jnp.mean(jnp.where(jnp.array(training_labels) == jnp.array(true_labels), 1, 0))

    # calculate the accuracy of the model policy
    for state, player, policy, true_optimal_action in zip(states, players, model_policy, true_optimal_actions_arr):
        # convert the policy to python list
        cc_state = cw.list_to_ccstate(list(state), player)
        model_action = get_action(cc, policy, cc_state, solver, "MovesForward")
        # true_label, stats, true_optimal_actions = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label
        model_policy_accuracy += calculate_policy_accuracy(model_action, true_optimal_action)

    
    # print("Iteration: {}, Number of states: {}, Model vs training: {}, Model vs true: {}, Training vs true: {}".format(iteration, len(data), model_vs_training_accuracy, model_vs_true_accuracy, training_vs_true_accuracy))
    result[int(iteration)] = { "accuracy": training_vs_true_accuracy, "training_label_accuracy": model_vs_training_accuracy, "ground_truth_label_accuracy": model_vs_true_accuracy, "training_policy_accuracy": training_policy_accuracy / len(states), "model_policy_accuracy": model_policy_accuracy / len(states) }
    

if __name__ == "__main__":
    model = evaluation.Evaluation()
    trainings_states = load_dict_of_states()
    result = {}

    global_data = {}
    for i in range(len(trainings_states)):
        global_data.update(trainings_states[str(i)])
        if i % 10 == 0:
            # get random 100000 states
            random_data = random.sample(global_data.items(), min(len(global_data), 100000))
            do_evaluation(i, random_data, result=result)
            print(result)
            # global_data = {}
            # exit()

    # sort result by iteration
    result = dict(sorted(result.items()))
    print(result)

    with open('result.pkl', 'wb') as fp:
        pickle.dump(result, fp)
    
    make_accuracy_graph()
            
            













    #     correct = 0
    #     ground_truth_label_correct = 0
    #     training_label_correct = 0
    #     total = 0

    #     states = []
    #     players = []
    #     training_policy = []
    #     training_labels = []
    #     policy_accuracy = 0
    #     true_policy = []
    #     true_labels = []


    #     for state, (player, policy, label) in iter_states.items():
    #         states.append(state)
    #         players.append(player)
    #         training_policy.append(policy)
    #         training_labels.append(label)

    #         cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate

    #         true_label, stats, true_optimal_actions = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label
    #         training_actions = get_action(cc, policy, cc_state, solver, "MovesForward") # get the action from the policy

    #         # calculate the accuracy of the policy
    #         policy_accuracy += calculate_policy_accuracy(training_actions, true_optimal_actions)
            
    #         true_labels.append(true_label)
    #         true_policy.append(stats)

    #     # evaluate the states in the iteration
    #     model_labels, model_policy = model.evaluate(state) # evaluate the state, get the label and policy from the model
    #     for model_label, training_label, true_label in zip(model_labels, training_labels, true_labels):
    #         if model_label == training_label:
    #             training_label_correct += 1
    #         if model_label == true_label:
    #             ground_truth_label_correct += 1
    #         if training_label == true_label:
    #             correct += 1

    #         total += 1
        
    #     print("Iteration: ", iteration)
    #     result[int(iteration)] = { "accuracy": correct / total, "training_label_accuracy": training_label_correct / total, "ground_truth_label_accuracy": ground_truth_label_correct / total, "policy_accuracy": policy_accuracy / total }
    
    # # sort result by iteration
    # result = dict(sorted(result.items()))

    # # save the result
    # with open('result.json', 'w') as fp:
    #     json.dump(result, fp)
    
    
    # # plot the result
    # make_accuracy_graph()