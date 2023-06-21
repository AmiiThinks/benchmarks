import os
import sys
# import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import random
import haiku as hk


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from evaluations import evaluation
from helpers import utils
from players import eval_nn_player as nnp

from mcts import mcts

solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)

BOARD_DIM = 5

def calculate_action_accuracy(model_dist, true_dist):
    model_action = np.argmax(model_dist)
    max_true_action = max(true_dist)
    acc = int(true_dist[model_action] == max_true_action)
    return acc

def make_accuracy_graph(result):
    iterations = list(result.keys())
    train_true_accuracy = [x["train_true_accuracy"] for x in result.values()]
    m_train_accuracy = [x["m_train_accuracy"] for x in result.values()]
    m_true_accuracy = [x["m_true_accuracy"] for x in result.values()]
    training_policy_accuracy = [x["training_policy_accuracy"] for x in result.values()]
    search_policy_accuracy = [x["search_policy_accuracy"] for x in result.values()]
    model_policy_accuracy = [x["model_policy_accuracy"] for x in result.values()]

    # Creating the first subplot for label accuracies
    plt.subplot(2, 1, 1)
    plt.plot(iterations, train_true_accuracy, label='Training comp. ground truth')
    # plt.plot(iterations, m_train_accuracy, label='Model comp. training labels')
    plt.plot(iterations, m_true_accuracy, label='Model comp. ground truth')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Label Accuracies')
    plt.legend()


    # Creating the second subplot for policy accuracy
    plt.subplot(2, 1, 2)
    plt.plot(iterations, training_policy_accuracy, label='Training Policy Accuracy')
    plt.plot(iterations, search_policy_accuracy, label='Search Policy Accuracy')
    plt.plot(iterations, model_policy_accuracy, label='Model Policy Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Policy Accuracy')
    plt.legend()

    # Adjusting the spacing between subplots
    plt.tight_layout()

    plt.savefig("../../results/accuracy_on_iteration.png")
    plt.close()
    return


def do_training_evaluation(model, nnpl, model_path, iteration, data):
    states = []
    players = []
    training_policy = []
    training_labels = []
    training_policy_accuracy = 0
    search_policy_accuracy = 0
    model_policy_accuracy = 0
    true_labels = []
    true_action_dist_arr = []

    for state, (player, train_policy, label) in data:
        states.append(state)
        players.append(player)
        training_policy.append(train_policy)
        training_labels.append(label)

        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        next_action, stats = nnpl.runMCTS(cc_state, 0) # run mcts on the state

        train_policy = jnp.array(train_policy)
        true_label, true_policy = utils.process_state_labels(mcts, cc, cc_state, solver, "MovesForward") # get the true label

        if cc_state.getToMove():
            search_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
            true_policy = pw.moves_distribution(true_policy, BOARD_DIM, reverse=True)
        else:
            search_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
            true_policy = pw.moves_distribution(true_policy, BOARD_DIM)

        training_action_dist = utils.get_action_distribution(cc, train_policy, cc_state) # get the action from the policy
        true_action_dist = utils.get_action_distribution(cc, true_policy, cc_state) # get the action from the true policy
        search_action_dist = utils.get_action_distribution(cc, search_dist, cc_state) # get the action from the search distribution

        true_action_dist_arr.append(true_action_dist) # append the true optimal actions

        training_policy_accuracy += calculate_action_accuracy(training_action_dist, true_action_dist) # calculate the accuracy of the training policy
        search_policy_accuracy += calculate_action_accuracy(search_action_dist, true_action_dist) # calculate the accuracy of the search policy

        true_labels.append(true_label)

    # evaluate the states in the iteration
    model_labels, model_policy = model.evaluate(states, model_path) # evaluate the state, get the label and policy from the model
    model_vs_training_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(training_labels), 1, 0))
    model_vs_true_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))
    training_vs_true_accuracy = jnp.mean(jnp.where(jnp.array(training_labels) == jnp.array(true_labels), 1, 0))

    # calculate the accuracy of the model policy
    for state, player, policy, true_action_dist in zip(states, players, model_policy, true_action_dist_arr):
        cc_state = cw.list_to_ccstate(list(state), player)
        model_action = utils.get_action_distribution(cc, policy, cc_state)
        model_policy_accuracy += calculate_action_accuracy(model_action, true_action_dist)

    search_policy_accuracy /= len(states)

    return model_vs_training_accuracy, model_vs_true_accuracy, training_vs_true_accuracy, training_policy_accuracy/len(states), search_policy_accuracy, model_policy_accuracy/len(states)

if __name__ == "__main__":

    model = evaluation.Evaluation()

    models_path = '/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mods'
    models = [f for f in os.listdir(models_path) if f.startswith('mod')]

    iterations = [int(model.split('_')[1]) for model in models]
    iter_models = dict(zip(iterations, models))
    iterations = sorted(iterations)

    trainings_states = pickle.load(open('../solver/iteration_training_states.p', 'rb'))
    result = {}

    global_data = {}
    print_res = ["Iteration", "Model comp. Training", "Model comp. True", "Training comp. True ", "Training Action Accuracy", "Search Action Accuracy", "Model Action Accuracy"]
    print('{:^25} {:^25} {:^25} {:^25} {:^25}  {:^25} {:^25}'.format(*print_res))
    for iteration in iterations:
        global_data.update(trainings_states[str(iteration)])
        if iteration % 10 == 0:
            model_path = os.path.join(models_path, iter_models[iteration])
            nnpl = nnp.NNPlayer(model_path, iteration)
            nn_player = mcts.MCTS(cc, nnpl)

            sampled_data = random.sample(global_data.items(), min(len(global_data), 100))
            m_train_accuracy, m_true_accuracy, train_true_accuracy, train_policy_accuracy, search_policy_accuracy, model_policy_accuracy = do_training_evaluation(model, nn_player, model_path, iteration, sampled_data)
            print('{:^25} {:^25.2f} {:^25.2f} {:^25.2f} {:^25.2f}  {:^25.2f} {:^25.2f}'.format(iteration, m_train_accuracy, m_true_accuracy, train_true_accuracy, train_policy_accuracy, search_policy_accuracy, model_policy_accuracy))
            result[iteration] = {"m_train_accuracy": m_train_accuracy, "m_true_accuracy": m_true_accuracy, "train_true_accuracy": train_true_accuracy, "training_policy_accuracy": train_policy_accuracy, "search_policy_accuracy": search_policy_accuracy, "model_policy_accuracy": model_policy_accuracy }
            
    # # sort result by iteration
    result = dict(sorted(result.items()))
    with open('training_evaluation_result.pkl', 'wb') as fp:
        pickle.dump(result, fp)

    result = pickle.load(open('training_evaluation_result.pkl', 'rb'))
    make_accuracy_graph(result)
    