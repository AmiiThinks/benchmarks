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
from players import eval_nn_player as nnp
from helpers import utils

from mcts import mcts

BOARD_DIM = 5
solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)


# load the training states dictionary
def load_states():
    # get all files inside neighbouring states folder
    files = os.listdir('../solver/neighboring_states/')
    states = {}
    for file in files:
        with open('../solver/neighboring_states/' + file, 'rb') as f:
            states.update(pickle.load(f))

    print("Loaded " + str(len(states)) + " states")
    return states

def make_accuracy_graph(result):
    iterations = list(result.keys())
    search_selection_acc = [x["search_selection_acc"] for x in result.values()]
    model_selection_acc = [x["model_selection_acc"] for x in result.values()]
    model_vs_true_outcome_accuracy = [
        x["model_vs_true_outcome_accuracy"] for x in result.values()
    ]

    # Creating the first subplot for label accuracies
    plt.subplot(2, 1, 1)
    plt.plot(iterations, model_vs_true_outcome_accuracy, label='Model vs True Label Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Label Accuracies')
    plt.legend()


    # Creating the second subplot for policy accuracy
    plt.subplot(2, 1, 2)
    plt.plot(iterations, search_selection_acc, label='Search Selection Accuracy')
    plt.plot(iterations, model_selection_acc, label='Model Selection Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Policy Accuracy')
    plt.legend()

    # Adjusting the spacing between subplots
    plt.tight_layout()

    plt.savefig("../../results/accuracy_on_random_iteration.png")
    plt.close()
    return


def do_evaluation(model, nnpl, model_path, iteration, iter_states):
    states = []
    cc_states = []
    players = []

    true_labels = []
    true_action_dist_arr = []

    search_selection_acc = 0
    model_selection_acc = 0

    for state, (player) in iter_states.items():
        states.append(state)
        players.append(player)

        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        cc_states.append(cc_state)
        next_action, stats = nnpl.runMCTS(cc_state, 0) # run mcts on the state
        true_label, true_stats = utils.process_state_labels(mcts, cc, cc_state, solver, "MovesForward") # get the true label
        true_labels.append(true_label)

        if cc_state.getToMove():
            search_dist = pw.moves_distribution(stats, BOARD_DIM, reverse=True)
            true_dist = pw.moves_distribution(true_stats, BOARD_DIM, reverse=True)
        else:
            search_dist = pw.moves_distribution(stats, BOARD_DIM)
            true_dist = pw.moves_distribution(true_stats, BOARD_DIM)
        
        search_action_dist = utils.get_action_distribution(cc, search_dist, cc_state) # get the action distribution from from search policy
        true_action_dist = utils.get_action_distribution(cc, true_dist, cc_state) # get the action distribution from the true policy

        true_action_dist_arr.append(true_action_dist)

        acc = utils.calculate_accuracy(search_action_dist, true_action_dist)
        search_selection_acc += acc

    model_labels, model_policy = model.evaluate(states, model_path) # evaluate the state, get the label and policy from the model
    for cc_state, m_policy, true_action_dist in zip(cc_states, model_policy, true_action_dist_arr):
        m_policy = m_policy.tolist()
        m_policy = np.array(m_policy[0])
        model_action_dist = utils.get_action_distribution(cc, m_policy, cc_state)

        model_acc = utils.calculate_accuracy(model_action_dist, true_action_dist)
        model_selection_acc += model_acc

    search_selection_acc = search_selection_acc / len(states)
    model_selection_acc = model_selection_acc / len(states)
    model_vs_true_outcome_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))

    return search_selection_acc, model_selection_acc, model_vs_true_outcome_accuracy

if __name__ == "__main__":

    model = evaluation.Evaluation()

    # models path
    models_path = '/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mods'
    models = [f for f in os.listdir(models_path) if f.startswith('mod')]

    iterations = [int(model.split('_')[1]) for model in models]
    iter_models = dict(zip(iterations, models))
    iterations = sorted(iterations)

    n_states = pickle.load(open('../solver/random_states.p', 'rb'))
    result = {}

    # global_data = {}
    # print_res = ["Iteration", "Search Action Selection Accuracy", "Model Action Selection Accuracy", "Model Label Accuracy"]
    # print('{:^25} {:^25} {:^25} {:^25}'.format(*print_res))
    # for iteration in iterations:
    #     global_data.update(n_states)
    #     if iteration % 10 == 0:
    #         model_path = os.path.join(models_path, iter_models[iteration])

    #         nnpl = nnp.NNPlayer(model_path, iteration)
    #         nn_player = mcts.MCTS(cc, nnpl)

    #         sampled_data = random.sample(global_data.items(), min(len(global_data), 100))
    #         sampled_data = dict(sampled_data)
    #         search_selection_acc, model_selection_acc, model_vs_true_outcome_accuracy = do_evaluation(model, nn_player, model_path, iteration, sampled_data)
    #         print('{:^25} {:^25.2f} {:^25.2f} {:^25.2f}'.format(iteration, search_selection_acc, model_selection_acc, model_vs_true_outcome_accuracy))
    #         result[iteration] = {'search_selection_acc': search_selection_acc, 'model_selection_acc': model_selection_acc, 'model_vs_true_outcome_accuracy': model_vs_true_outcome_accuracy}
            
    # # # sort result by iteration
    # result = dict(sorted(result.items()))
    # with open('random_evaluation_result.pkl', 'wb') as fp:
    #     pickle.dump(result, fp)

    result = pickle.load(open('random_evaluation_result.pkl', 'rb'))
    make_accuracy_graph(result)