import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import random
import multiprocessing as mp
from multiprocessing import set_start_method


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from players import eval_nn_player as nnp
from helpers import log_helper, utils
import evaluation

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

from mcts import mcts

BOARD_DIM = 5
# solved_data = "/data/home/bkarki/solved_data/"
solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
solver = cw.Solver(solved_data, True, False)

# init logger
# logger_path = "/data/home/bkarki/cc/cc_556/chinese-checkers/python2/evaluations/"
logger_path = "/Users/bigyankarki/Desktop/bigyan/cc/chinese_checkers_5_5_6/python2/evaluations"
logger = log_helper.setup_logger('eval_search', logger_path, '../logs/eval_search.log')

def do_training_evaluation(model, nnpl, model_path, iteration, iter_states):
    states = []
    cc_states = []
    players = []

    train_labels = []
    true_labels = []
    true_policy = []
    true_action_dist_arr = []

    train_selection_acc = 0
    search_selection_acc = 0
    model_selection_acc = 0

    for state, (player, train_policy, train_label) in iter_states.items():
        states.append(state)
        players.append(player)
        train_labels.append(train_label)

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
        
        true_policy.append(true_dist)
        train_action_dist = utils.get_action_distribution(cc, train_policy, cc_state) # get the action distribution of train_policy
        search_action_dist = utils.get_action_distribution(cc, search_dist, cc_state) # get the action distribution from from search policy
        true_action_dist = utils.get_action_distribution(cc, true_dist, cc_state) # get the action distribution from the true policy

        true_action_dist_arr.append(true_action_dist)

        acc = utils.calculate_accuracy(search_action_dist, true_action_dist)
        search_selection_acc += acc

        acc = utils.calculate_accuracy(train_action_dist, true_action_dist)
        train_selection_acc += acc

    # get the model action distribution
    model_labels, model_policy = model.evaluate(states, model_path) # evaluate the state, get the label and policy from the model
    for state, m_policy, true_action_dist in zip(cc_states, model_policy, true_action_dist_arr):
        
        m_policy = m_policy.tolist()
        m_policy = np.array(m_policy[0])
        model_action_dist = utils.get_action_distribution(cc, m_policy, state)

        model_acc = utils.calculate_accuracy(model_action_dist, true_action_dist)
        model_selection_acc += model_acc

    search_selection_acc = search_selection_acc / len(states)
    model_selection_acc = model_selection_acc / len(states)
    train_selection_acc = train_selection_acc / len(states)

    model_vs_true_outcome_accuracy = jnp.mean(jnp.where(model_labels == jnp.array(true_labels), 1, 0))
    train_vs_true_outcome_accuracy = jnp.mean(jnp.where(jnp.array(train_labels) == jnp.array(true_labels), 1, 0))

    return search_selection_acc, model_selection_acc, train_selection_acc, model_vs_true_outcome_accuracy, train_vs_true_outcome_accuracy


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


# define each process task for multiprocessing
def process_iteration(iteration, iter_models, models_path, training_states, neighboring_states, random_states):
    model = evaluation.Evaluation()
    model_path = os.path.join(models_path, iter_models[iteration])
    nnpl = nnp.NNPlayer(model_path, iteration)
    nn_player = mcts.MCTS(cc, nnpl)

    print("Iteration {} is running on process {}".format(iteration, os.getpid()))
    t_search_action_accuracy, t_model_action_accuracy, t_train_action_accuracy, t_model_true_outcome_accuracy, t_train_true_outcome_accuracy = do_training_evaluation(model, nn_player, model_path, iteration, training_states)
    n_search_action_accuracy, n_model_action_accuracy, n_model_true_outcome_accuracy = do_evaluation(model, nn_player, model_path, iteration, neighboring_states)
    r_search_action_accuracy, r_model_action_accuracy, r_model_true_outcome_accuracy = do_evaluation(model, nn_player, model_path, iteration, random_states)
    return iteration, {'t_search_action_accuracy': t_search_action_accuracy, 't_model_action_accuracy': t_model_action_accuracy, 't_train_action_accuracy': t_train_action_accuracy, 't_model_true_outcome_accuracy': t_model_true_outcome_accuracy, 't_train_true_outcome_accuracy': t_train_true_outcome_accuracy, 'n_search_action_accuracy': n_search_action_accuracy, 'n_model_action_accuracy': n_model_action_accuracy, 'n_model_true_outcome_accuracy': n_model_true_outcome_accuracy, 'r_search_action_accuracy': r_search_action_accuracy, 'r_model_action_accuracy': r_model_action_accuracy, 'r_model_true_outcome_accuracy': r_model_true_outcome_accuracy}


if __name__ == "__main__":
    set_start_method('spawn') # this is required for multiprocessing to work properly on linux

    # model = evaluation.Evaluation()
    training_states = pickle.load(open("../solver/sampled_training_states.p", "rb"))
    neighboring_states = pickle.load(open("../solver/sampled_neighboring_states.p", "rb"))
    random_states = pickle.load(open("../solver/random_states.p", "rb"))

    # sample random 10 states from each set
    n_samples = 100
    training_states = random.sample(training_states.items(), n_samples)
    neighboring_states = random.sample(neighboring_states.items(), n_samples)
    random_states = random.sample(random_states.items(), n_samples)

    training_states = dict(training_states)
    neighboring_states = dict(neighboring_states)
    random_states = dict(random_states)

    # models path
    # models_path = '/data/home/bkarki/cc/results/async_jax_with_19_res_blocks/trained_models/'
    models_path = '/Users/bigyankarki/Desktop/bigyan/results/19_res_blocks/mods'
    models = [f for f in os.listdir(models_path) if f.startswith('mod')]
    
    iterations = [int(model.split('_')[1]) for model in models]
    iter_models = dict(zip(iterations, models))
    iterations = [i for i in iterations if i % 10 == 0] # only take every 10th iteration
    iterations = sorted(iterations)
    result = dict()

    print_res = ["Iteration", "Training Search Accuracy", "Training Model Accuracy", "Training Train Accuracy", "Training Model True Outcome Accuracy", "Training Train True Outcome Accuracy", "Neighboring Search Accuracy", "Neighboring Model Accuracy", "Neigboring Model True Outcome Accuracy", "Random Search Accuracy", "Random Model Accuracy", "Random Model True Outcome Accuracy"]
    logger.info('{:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36}'.format(*print_res))

    pool = mp.Pool(mp.cpu_count())
    for iteration, res in pool.starmap(process_iteration, [(iteration, iter_models, models_path, training_states, neighboring_states, random_states) for iteration in iterations]):
        result[iteration] = res
        print_res = [iteration, res['t_search_action_accuracy'], res['t_model_action_accuracy'], res['t_train_action_accuracy'], res['t_model_true_outcome_accuracy'], res['t_train_true_outcome_accuracy'], res['n_search_action_accuracy'], res['n_model_action_accuracy'], res['n_model_true_outcome_accuracy'], res['r_search_action_accuracy'], res['r_model_action_accuracy'], res['r_model_true_outcome_accuracy']]
        logger.info('{:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36}'.format(*print_res))

    pool.close()
    pool.join()

    pickle.dump(result, open('search_policy_results_128.p', 'wb'))

    result = pickle.load(open('search_policy_results_128.p', 'rb'))
    # other_res = pickle.load(open('eval_results_on_ground_truth_t.p', 'rb'))
    utils.plot_state_evaluation(result, sample_size=1000)

    # for iteration in iterations:
    #     if iteration % 10 == 0:
    #         iteration, res = process_iteration(iteration, iter_models, models_path, training_states, neighboring_states, random_states)
    #         result[iteration] = res

    #         print_res = [iteration, res['t_search_action_accuracy'], res['t_model_action_accuracy'], res['t_train_action_accuracy'], res['t_model_true_outcome_accuracy'], res['t_train_true_outcome_accuracy'], res['n_search_action_accuracy'], res['n_model_action_accuracy'], res['n_model_true_outcome_accuracy'], res['r_search_action_accuracy'], res['r_model_action_accuracy'], res['r_model_true_outcome_accuracy']]
    #         logger.info('{:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36} | {:^36}'.format(*print_res))

    #         pickle.dump(result, open("search_policy_results.p", "wb"))
    #         utils.plot_state_evaluation(result, sample_size=n_samples)

    # result = pickle.load(open("search_policy_results.p", "rb"))
    # utils.plot_state_evaluation(result, sample_size=n_samples)










   