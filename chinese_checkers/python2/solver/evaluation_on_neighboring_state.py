import os
import sys
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from evaluations import evaluation

from mcts import mcts


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


def get_action(cc, policy, state, solver, move_type):
    legals, legal_moves = remove_illegal('MovesForward', policy, state, cc, state.getToMove())
    legals = jnp.array([legals])
    washed = jnp.where(jnp.array(legals).astype(bool), policy, -1e32*jnp.ones_like(policy))
    washed = jax.nn.softmax(washed)
    probs = np.array(get_probabilities(washed, legal_moves, state.getToMove()))
    
    # select move with highest probability
    move_ind = np.argwhere(probs == np.amax(probs)).flatten()

    # convert index of probs to 1 for those in move_ind
    probs = np.zeros_like(probs)
    probs[move_ind] = 1.0

    # print(probs)
    # move = legal_moves[move_ind]

    # print state representation
    # print("From state : " + str(state.getBoard()))
    # cc.ApplyMove(state, move)
    # print("To state : " + str(state.getBoard()))

    # print("Move: " + str(move_ind) + " " + str(move.getFrom()) + " " + str(move.getTo()))
    # print(move.getFrom(), move.getTo(), probs[move_ind], state.getToMove())
    
    return probs

def calculate_policy_accuracy(training_policy, ground_truth_policy):
    # print(training_policy)
    # print(ground_truth_policy)
    # return 1 if index if index value is 1 for both policies
    return 1 if np.sum(np.multiply(training_policy, ground_truth_policy)) > 0 else 0




# load the training states dictionary
def load_dict_of_states():
    # get all files inside neighbouring states folder
    files = os.listdir('neighbouring_states')
    states = {}
    for file in files:
        with open('neighbouring_states/' + file, 'rb') as f:
            states.update(pickle.load(f))

    print("Loaded " + str(len(states)) + " states")
    return states

def load_result_data():
    data =  pickle.load(open('neighbours_eval_result.pkl', 'rb'))
    return data

def make_accuracy_graph():
    result = load_result_data()
    iterations = list(result.keys())
    ground_truth_label_accuracy = [x["ground_truth_label_accuracy"] for x in result.values()]
    # policy_accuracy = [x["policy_accuracy"] for x in result.values()]

    plt.plot(iterations, ground_truth_label_accuracy, label="Ground Truth Label Accuracy")
    # plt.plot(iterations, policy_accuracy, label="Policy Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("neighboring_states_accuracy.png")
    plt.close()
    return


def do_evaluation(iteration, iter_states, result):
    states = []
    players = []
    policy_accuracy = 0
    true_policy = []
    true_labels = []

    data = []

    for state, (player) in iter_states.items():
        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        true_label, stats, true_optimal_actions = process_state_labels(cc, cc_state, solver, "MovesForward") # get the true label
        # training_actions = get_action(cc, policy, cc_state, solver, "MovesForward") # get the action from the policy

        # calculate the accuracy of the policy
        # policy_accuracy += calculate_policy_accuracy(training_actions, true_optimal_actions)
        data.append((state, player, stats, true_label))
        
    
    # get random states from the data
    random_states = random.sample(data, min(len(data), 100000))
    for state, player, stats, true_label in random_states:
        states.append(state)
        players.append(player)
        true_labels.append(true_label)
        true_policy.append(stats)
    
    print("Evaluating the model in {} states out of {} states".format(len(states), len(iter_states)))
    # evaluate the states in the iteration
    model_labels, model_policy = model.evaluate(states) # evaluate the state, get the label and policy from the model
    model_vs_true_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))
    
    print("Iteration: {}, Number of states: {}, Model vs true: {}".format(iteration, len(iter_states), model_vs_true_accuracy))
    result[int(iteration)] = {"ground_truth_label_accuracy": model_vs_true_accuracy, "policy_accuracy": 0 }
    

if __name__ == "__main__":
    model = evaluation.Evaluation()
    states = load_dict_of_states()
    result = {}

    global_data = {}
    for i in range(len(states)):
        global_data.update(states[str(i)])
        if i % 10 == 0:
            do_evaluation(i, global_data, result=result)
            # global_data = {}

            print(result)
            # exit()

    # sort result by iteration
    result = dict(sorted(result.items()))
    print(result)

    # save the result
    with open('neighbours_eval_result.pkl', 'wb') as fp:
        pickle.dump(result, fp)
    
    
    # # plot the result
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