import numpy as np
import os
from os.path import exists
import sys
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import pickle
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

def remove_illegal(move_type, policy, state, cc, reverse=False):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)

    legals = pw.moves_existence(legal_moves, reverse)
    washed = jnp.array(legals) * policy
    washed = washed / jnp.sum(washed)
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

def process_state_labels(mcts, cc, state, solver, move_type):

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

# create .npy file to store data [state, player, sample_dist, outcome], for parallel self play
def add_to_buffer(states, outcomes, buffer_q, config, self_play_iteration, game_count):
    state = []
    player = []
    sample_dist = []
    outcome = []

    for i in range(len(states)):
        state.append(states[i][0])
        player.append(states[i][1])
        sample_dist.append(states[i][2])
        outcome.append(outcomes[i])
        
        data = (states[i][0], states[i][1], states[i][2], outcomes[i]) # state, player, policy, outcome

        # put to training_q
        buffer_q.push(data)

    res = [state, player, sample_dist, outcome]
    # print("Added to buffer queue", np.array(res, dtype=object).shape)

    # save to file
    # output_folder = os.path.join(config["file"]["training_data_path"], str("iteration_{}".format(self_play_iteration)))
    # if not exists(output_folder):
    #   os.mkdir(output_folder)

    # file_name = str("game_{}.npy".format(game_count))
    # output_path = os.path.join(output_folder, file_name)
    # np.save(output_path, np.array(res, dtype=object))

def create_dataset_from_buffer_q(samples):
  states = []
  players = []
  outcomes = []
  policy = []

  for sample in samples:
    states.append(sample[0])
    players.append(sample[1])
    policy.append(sample[2])
    outcomes.append(sample[3])

  states = jnp.array(states, dtype=jnp.float32)
  players = jnp.array(players, dtype=jnp.float32)
  players = jnp.reshape(players, (-1, players.shape[0]))
  outcomes = jnp.array(outcomes, dtype=jnp.float32)
  outcomes = jnp.reshape(outcomes, (-1, outcomes.shape[0]))
  policy = jnp.array(policy, dtype=jnp.float32)

  return states, players, policy, outcomes

# plot the training process
def plot(path, iteration, total_loss, value_loss, policy_loss, self_play_iter):
    # plot the loss for each epoch and add epch as legend'
    plt.plot(iteration, total_loss, label='total loss')
    plt.plot(iteration, value_loss, label='value loss')
    plt.plot(iteration, policy_loss, label='policy loss')
    plt.legend()
    plt.xlabel('Iteration {}'.format(self_play_iter))
    plt.ylabel('Loss')
    plt.title('Iteration vs Loss')
    path = os.path.join(path, 'train_iteration_{}.png'.format(self_play_iter))
    plt.savefig(path)
    plt.close()
    plt.clf()

# save model
def save_model(path, params):
  params = jax.device_get(params)
  with open(path, 'wb') as fp:
    pickle.dump(params, fp)

# save optimizer state
def save_optimizer_state(path, opt_state):
  opt_state = jax.device_get(opt_state)
  with open(path, 'wb') as fp:
    pickle.dump(opt_state, fp)

# load model
def load_model_params(path):
  with open(path, 'rb') as fp:
      params = pickle.load(fp)
  return jax.device_put(params)

# load optimizer state
def load_optimizer_state(path):
  with open(path, 'rb') as fp:
      opt_state = pickle.load(fp)
  return jax.device_put(opt_state)

# get latest model parameters
def get_latest_model():
  model = np.load('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python/models/latest_model.npy', allow_pickle=True)[()]
  return model

# save total loss in a .npy file
def plot_loss(path, self_play_iter, loss_arr, iteration_arr, name):
  # path = '/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/plots/parallel_play/{}.pkl'.format(name)
  pkl_file_name = '{}.pkl'.format(name)
  plt_file_name = '{}.png'.format(name)
  pkl_path = os.path.join(path, pkl_file_name)
  plt_path = os.path.join(path, plt_file_name)
  
  data = {self_play_iter: [loss_arr, iteration_arr]}
  if self_play_iter == 0:
    with open(pkl_path, 'wb') as fp:
      pickle.dump(data, fp)
  else:
    with open(pkl_path, 'rb') as fp:
      prev_data = pickle.load(fp)
    prev_data.update(data)

    # plot total loss
    for key in prev_data.keys():
      plt.plot(prev_data[key][1], prev_data[key][0], label='{}'.format(key))

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('{} loss'.format(name.split('_')[0]))
    plt.title('Iteration vs {}'.format(name))
    plt.savefig(plt_path)
    plt.close()

    with open(pkl_path, 'wb') as fp:
      pickle.dump(prev_data, fp)

def plot_losses_in_single_line(path):
  # load total loss, value loss, policy loss from .pkl file
  total_loss_path = os.path.join(path, 'total_loss.pkl')
  value_loss_path = os.path.join(path, 'value_loss.pkl')
  policy_loss_path = os.path.join(path, 'policy_loss.pkl')
  single_loss_line = os.path.join(path, 'single_loss_line.png')
  with open(total_loss_path, 'rb') as fp:
    total_loss = pickle.load(fp)
  with open(value_loss_path, 'rb') as fp:
    value_loss = pickle.load(fp)
  with open(policy_loss_path, 'rb') as fp:
    policy_loss = pickle.load(fp)

  def _update_losses(loss_arr):
    iteration = 0
    updated_total_loss_arr = []
    updated_iteration_arr = []
    for key in loss_arr.keys():
      arr = loss_arr[key][0]
      
      updated_total_loss_arr.extend(arr)
      for i in range(len(loss_arr[key][0])):
        updated_iteration_arr.append(iteration)
        iteration += 1

    return updated_total_loss_arr, updated_iteration_arr

  total_loss_arr, iteration_arr = _update_losses(total_loss)
  value_loss_arr, _ = _update_losses(value_loss)
  policy_loss_arr, _ = _update_losses(policy_loss)
  # print(len(total_loss_arr), len(iteration_arr))

  plt.plot(iteration_arr, total_loss_arr, label='total loss')
  plt.plot(iteration_arr, value_loss_arr, label='value loss')
  plt.plot(iteration_arr, policy_loss_arr, label='policy loss')
  plt.legend()
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Iteration vs Loss')
  plt.savefig(single_loss_line)
  plt.close()

# plot for iteration stats
def plot_iteration_stats(config, iteration_stats):
  path = config['file']["training_plots_path"]
  pkl_file_name = '{}.pkl'.format("iteration_stats")
  pkl_path = os.path.join(path, pkl_file_name)

  with open(pkl_path, 'wb') as fp:
    pickle.dump(iteration_stats, fp)


  iterations = []
  average_times = []
  game_counts = []
  average_states = []
  player_1_wins = []
  player_2_wins = []
  draws = []
  training_times = []
  
  for iteration in list(iteration_stats.keys())[:-1]:
      iterations.append(iteration)
      average_times.append(iteration_stats[iteration]["average_time"])
      game_counts.append(iteration_stats[iteration]["game_count"])
      average_states.append(iteration_stats[iteration]["average_states"])
      player_1_wins.append(iteration_stats[iteration]["player_win_status"][1])
      player_2_wins.append(iteration_stats[iteration]["player_win_status"][2])
      draws.append(iteration_stats[iteration]["player_win_status"][0])
      training_times.append(iteration_stats[iteration]["training_time"])

  fig, ax = plt.subplots(2, 2, figsize=(20, 10))
  ax[0, 0].plot(iterations, average_times)
  ax[0, 0].set_title("Average Time")
  ax[0, 0].set_xlabel("Iteration")
  ax[0, 0].set_ylabel("Average Time (s)")

  ax[0, 1].plot(iterations, game_counts)
  ax[0, 1].set_title("Game Count")
  ax[0, 1].set_xlabel("Iteration")
  ax[0, 1].set_ylabel("Game Count")

  ax[1, 0].plot(iterations, average_states)
  ax[1, 0].set_title("Average States")
  ax[1, 0].set_xlabel("Iteration")
  ax[1, 0].set_ylabel("Average States")

  ax[1, 1].plot(iterations, player_1_wins, label="Player 1 Wins")
  ax[1, 1].plot(iterations, player_2_wins, label="Player 2 Wins")
  ax[1, 1].plot(iterations, draws, label="Draws")
  ax[1, 1].set_title("Player Win Status")
  ax[1, 1].set_xlabel("Iteration")
  ax[1, 1].set_ylabel("Win Status")
  ax[1, 1].legend()

  path = os.path.join(config["file"]["training_plots_path"], "stats.png")
  plt.savefig(path)
  plt.close()

  fig, ax = plt.subplots(1, 1, figsize=(20, 10))
  ax.plot(iterations, training_times)
  ax.set_title("Training Time")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Training Time (s)")
  path = os.path.join(config["file"]["training_plots_path"], "training_stats.png")
  plt.savefig(path)
  plt.close()


def create_dataset_from_folder(folder_path):
  # get all the files in the folder
  files = os.listdir(folder_path)

  # get all the files in each iteration folder
  iteration_folders = []
  for file in files:
    if file.startswith("iteration"):
      iteration_folders.append(os.path.join(folder_path, file))

  # get all the files in each iteration folder
  iteration_files = []
  for iteration_folder in iteration_folders:
    files = os.listdir(iteration_folder)
    iteration_files.append(files)

  # get all the files in each iteration folder
  iteration_files_path = []
  for iteration_folder in iteration_folders:
    files = os.listdir(iteration_folder)
    for file in files:
      iteration_files_path.append(os.path.join(iteration_folder, file))

  # get all the data from each file
  states = []
  players = []
  outcomes = []
  policy = []
  tuple_data = []
  batch_size = 1000

  for file in iteration_files_path:
    data = np.load(file, allow_pickle=True)
    tuple_data.append((data[0], data[1], data[2], data[3])) # state, player, policy, outcome

  # shuffle the data
  # np.random.shuffle(tuple_data)

  # make a dictionary of the data, where the key is state and value is the rest of the data
  original_data = []
  data_dict = {}
  for states, players, policy, outcomes in tuple_data:
    for state, player, policy, outcome in zip(states, players, policy, outcomes):
      original_data.append((state, player, policy, outcome))
      data_dict[tuple(state)] = (player, policy, outcome)

  return original_data[:1000]


      
  # print("Number of original states: {}".format(len(original_data)))
  # print("Number of unique states: {}".format(len(data_dict.keys())))
  # get the unique states
  

  # create dataset of batch size
  # for sample in tuple_data:
  #   states.extend(sample[0])
  #   players.extend(sample[1])
  #   policy.extend(sample[2])
  #   outcomes.extend(sample[3])

  # states = jnp.array(states, dtype=jnp.float32)[:batch_size]
  # players = jnp.array(players, dtype=jnp.float32)[:batch_size]
  # players = jnp.reshape(players, (-1, players.shape[0]))
  # outcomes = jnp.array(outcomes, dtype=jnp.float32)[:batch_size]
  # outcomes = jnp.reshape(outcomes, (-1, outcomes.shape[0]))
  # policy = jnp.array(policy, dtype=jnp.float32)[:batch_size]

  # return states, players, policy, outcomes

# calculate confidence interval
def calculate_confidence_interval(mean, sample_size, confidence):
    z = 1.96 # z-score for 95% confidence
    std_err = math.sqrt(mean * (1 - mean) / sample_size) # get standard error
    margin_of_error = z * std_err # get margin of error
    confidence_interval = (mean - margin_of_error, mean + margin_of_error) # get confidence interval
    return confidence_interval

def plot_state_evaluation(result, sample_size):
    iterations = list(result.keys())

    # actions accuracy
    t_search_action_accuracy = [result[iteration]['t_search_action_accuracy'] for iteration in iterations]
    t_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_search_action_accuracy]

    n_search_action_accuracy = [result[iteration]['n_search_action_accuracy'] for iteration in iterations]
    n_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_search_action_accuracy]

    r_search_action_accuracy = [result[iteration]['r_search_action_accuracy'] for iteration in iterations]
    r_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_search_action_accuracy]

    t_model_action_accuracy = [result[iteration]['t_model_action_accuracy'] for iteration in iterations]
    t_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_action_accuracy]

    n_model_action_accuracy = [result[iteration]['n_model_action_accuracy'] for iteration in iterations]
    n_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_model_action_accuracy]

    r_model_action_accuracy = [result[iteration]['r_model_action_accuracy'] for iteration in iterations]
    r_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_model_action_accuracy]

    # t_train_action_accuracy = [result[iteration]['t_train_action_accuracy'] for iteration in iterations]
    # t_train_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_train_action_accuracy]

    # t_model_true_outcome_accuracy = [result[iteration]['t_model_true_outcome_accuracy'] for iteration in iterations]
    # t_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_true_outcome_accuracy]

    # t_train_true_outcome_accuracy = [result[iteration]['t_train_true_outcome_accuracy'] for iteration in iterations]
    # t_train_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_train_true_outcome_accuracy]

    # n_model_true_outcome_accuracy = [result[iteration]['n_model_true_outcome_accuracy'] for iteration in iterations]
    # n_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_model_true_outcome_accuracy]

    # r_model_true_outcome_accuracy = [result[iteration]['r_model_true_outcome_accuracy'] for iteration in iterations]
    # r_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_model_true_outcome_accuracy]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # subplot 1: Training state accuracies
    axs[0, 0].plot(iterations, t_search_action_accuracy, label='Search Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_search_action_accuracy_cf], [x[1] for x in t_search_action_accuracy_cf], alpha=0.2)

    axs[0, 0].plot(iterations, t_model_action_accuracy, label='Model Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_model_action_accuracy_cf], [x[1] for x in t_model_action_accuracy_cf], alpha=0.2)

    # axs[0, 0].plot(iterations, t_train_action_accuracy, label='Train Accuracy')
    # axs[0, 0].fill_between(iterations, [x[0] for x in t_train_action_accuracy_cf], [x[1] for x in t_train_action_accuracy_cf], alpha=0.2)
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Action Accuracy')
    axs[0, 0].set_title('Training State')
    axs[0, 0].legend()

    # subplot 2: Neighboring state accuracies
    axs[0, 1].plot(iterations, n_search_action_accuracy, label='Search Accuracy')
    axs[0, 1].fill_between(iterations, [x[0] for x in n_search_action_accuracy_cf], [x[1] for x in n_search_action_accuracy_cf], alpha=0.2)

    axs[0, 1].plot(iterations, n_model_action_accuracy, label='Model Accuracy')
    axs[0, 1].fill_between(iterations, [x[0] for x in n_model_action_accuracy_cf], [x[1] for x in n_model_action_accuracy_cf], alpha=0.2)
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Action Accuracy')
    axs[0, 1].set_title('Neighboring State')
    axs[0, 1].legend()

    # subplot 3: Random state accuracies
    axs[1, 0].plot(iterations, r_search_action_accuracy, label='Search Accuracy')
    axs[1, 0].fill_between(iterations, [x[0] for x in r_search_action_accuracy_cf], [x[1] for x in r_search_action_accuracy_cf], alpha=0.2)

    axs[1, 0].plot(iterations, r_model_action_accuracy, label='Model Accuracy')
    axs[1, 0].fill_between(iterations, [x[0] for x in r_model_action_accuracy_cf], [x[1] for x in r_model_action_accuracy_cf], alpha=0.2)
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('Action Accuracy')
    axs[1, 0].set_title('Random State')
    axs[1, 0].legend()

    # subplot 4: All state accuracies
    axs[1, 1].plot(iterations, t_search_action_accuracy, label='Training Search Accuracy')
    axs[1, 1].plot(iterations, n_search_action_accuracy, label='Neighboring Search Accuracy')
    axs[1, 1].plot(iterations, r_search_action_accuracy, label='Random Search Accuracy')
    axs[1, 1].plot(iterations, t_model_action_accuracy, label='Training Model Accuracy')
    axs[1, 1].plot(iterations, n_model_action_accuracy, label='Neighboring Model Accuracy')
    axs[1, 1].plot(iterations, r_model_action_accuracy, label='Random Model Accuracy')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Action Accuracy')
    axs[1, 1].set_title('All States')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('../../results/search_accuracy.png')
    plt.clf()

    # plot outcome accuracy
    # plt.plot(iterations, t_model_true_outcome_accuracy, label='Training Model Accuracy')
    # plt.fill_between(iterations, [x[0] for x in t_model_true_outcome_accuracy_cf], [x[1] for x in t_model_true_outcome_accuracy_cf], alpha=0.2)

    # plt.plot(iterations, t_train_true_outcome_accuracy, label='Training Train Accuracy')
    # plt.fill_between(iterations, [x[0] for x in t_train_true_outcome_accuracy_cf], [x[1] for x in t_train_true_outcome_accuracy_cf], alpha=0.2)

    # plt.plot(iterations, n_model_true_outcome_accuracy, label='Neighboring Model Accuracy')
    # plt.fill_between(iterations, [x[0] for x in n_model_true_outcome_accuracy_cf], [x[1] for x in n_model_true_outcome_accuracy_cf], alpha=0.2)

    # plt.plot(iterations, r_model_true_outcome_accuracy, label='Random Model Accuracy')
    # plt.fill_between(iterations, [x[0] for x in r_model_true_outcome_accuracy_cf], [x[1] for x in r_model_true_outcome_accuracy_cf], alpha=0.2)


    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.title('Outcome Accuracy')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('../../results/outcome_accuracy.png')
    # plt.clf()


def plot_state_evaluation_test(result, other_result, sample_size):
    iterations = list(result.keys())

    # actions accuracy
    t_search_action_accuracy = [result[iteration]['t_search_action_accuracy'] for iteration in iterations]
    t_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_search_action_accuracy]

    n_search_action_accuracy = [result[iteration]['n_search_action_accuracy'] for iteration in iterations]
    n_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_search_action_accuracy]

    r_search_action_accuracy = [result[iteration]['r_search_action_accuracy'] for iteration in iterations]
    r_search_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_search_action_accuracy]

    t_model_action_accuracy = [other_result[iteration]['t_action_accuracy'] for iteration in iterations]
    t_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_action_accuracy]

    n_model_action_accuracy = [other_result[iteration]['n_action_accuracy'] for iteration in iterations]
    n_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_model_action_accuracy]

    r_model_action_accuracy = [other_result[iteration]['r_action_accuracy'] for iteration in iterations]
    r_model_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_model_action_accuracy]

    t_train_action_accuracy = [other_result[iteration]['t_train_action_accuracy'] for iteration in iterations]
    t_train_action_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_train_action_accuracy]

    t_model_true_outcome_accuracy = [result[iteration]['t_model_true_outcome_accuracy'] for iteration in iterations]
    t_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_model_true_outcome_accuracy]

    t_train_true_outcome_accuracy = [result[iteration]['t_train_true_outcome_accuracy'] for iteration in iterations]
    t_train_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in t_train_true_outcome_accuracy]

    n_model_true_outcome_accuracy = [result[iteration]['n_model_true_outcome_accuracy'] for iteration in iterations]
    n_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in n_model_true_outcome_accuracy]

    r_model_true_outcome_accuracy = [result[iteration]['r_model_true_outcome_accuracy'] for iteration in iterations]
    r_model_true_outcome_accuracy_cf = [calculate_confidence_interval(acc, sample_size, 0.95) for acc in r_model_true_outcome_accuracy]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # subplot 1: Training state accuracies
    axs[0, 0].plot(iterations, t_search_action_accuracy, label='Search Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_search_action_accuracy_cf], [x[1] for x in t_search_action_accuracy_cf], alpha=0.2)

    axs[0, 0].plot(iterations, t_model_action_accuracy, label='Model Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_model_action_accuracy_cf], [x[1] for x in t_model_action_accuracy_cf], alpha=0.2)

    axs[0, 0].plot(iterations, t_train_action_accuracy, label='Train Accuracy')
    axs[0, 0].fill_between(iterations, [x[0] for x in t_train_action_accuracy_cf], [x[1] for x in t_train_action_accuracy_cf], alpha=0.2)
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Action Accuracy')
    axs[0, 0].set_title('Training State')
    axs[0, 0].legend()

    # subplot 2: Neighboring state accuracies
    axs[0, 1].plot(iterations, n_search_action_accuracy, label='Search Accuracy')
    axs[0, 1].fill_between(iterations, [x[0] for x in n_search_action_accuracy_cf], [x[1] for x in n_search_action_accuracy_cf], alpha=0.2)

    axs[0, 1].plot(iterations, n_model_action_accuracy, label='Model Accuracy')
    axs[0, 1].fill_between(iterations, [x[0] for x in n_model_action_accuracy_cf], [x[1] for x in n_model_action_accuracy_cf], alpha=0.2)
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Action Accuracy')
    axs[0, 1].set_title('Neighboring State')
    axs[0, 1].legend()

    # subplot 3: Random state accuracies
    axs[1, 0].plot(iterations, r_search_action_accuracy, label='Search Accuracy')
    axs[1, 0].fill_between(iterations, [x[0] for x in r_search_action_accuracy_cf], [x[1] for x in r_search_action_accuracy_cf], alpha=0.2)

    axs[1, 0].plot(iterations, r_model_action_accuracy, label='Model Accuracy')
    axs[1, 0].fill_between(iterations, [x[0] for x in r_model_action_accuracy_cf], [x[1] for x in r_model_action_accuracy_cf], alpha=0.2)
    axs[1, 0].set_xlabel('Iterations')
    axs[1, 0].set_ylabel('Action Accuracy')
    axs[1, 0].set_title('Random State')
    axs[1, 0].legend()

    # subplot 4: All state accuracies
    axs[1, 1].plot(iterations, t_search_action_accuracy, label='Training Search Accuracy')
    axs[1, 1].plot(iterations, n_search_action_accuracy, label='Neighboring Search Accuracy')
    axs[1, 1].plot(iterations, r_search_action_accuracy, label='Random Search Accuracy')
    axs[1, 1].plot(iterations, t_model_action_accuracy, label='Training Model Accuracy')
    axs[1, 1].plot(iterations, n_model_action_accuracy, label='Neighboring Model Accuracy')
    axs[1, 1].plot(iterations, r_model_action_accuracy, label='Random Model Accuracy')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Action Accuracy')
    axs[1, 1].set_title('All States')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('../../results/search_accuracy.png')
    plt.clf()

    # plot outcome accuracy
    plt.plot(iterations, t_model_true_outcome_accuracy, label='Training Model Accuracy')
    plt.fill_between(iterations, [x[0] for x in t_model_true_outcome_accuracy_cf], [x[1] for x in t_model_true_outcome_accuracy_cf], alpha=0.2)

    plt.plot(iterations, t_train_true_outcome_accuracy, label='Training Train Accuracy')
    plt.fill_between(iterations, [x[0] for x in t_train_true_outcome_accuracy_cf], [x[1] for x in t_train_true_outcome_accuracy_cf], alpha=0.2)

    plt.plot(iterations, n_model_true_outcome_accuracy, label='Neighboring Model Accuracy')
    plt.fill_between(iterations, [x[0] for x in n_model_true_outcome_accuracy_cf], [x[1] for x in n_model_true_outcome_accuracy_cf], alpha=0.2)

    plt.plot(iterations, r_model_true_outcome_accuracy, label='Random Model Accuracy')
    plt.fill_between(iterations, [x[0] for x in r_model_true_outcome_accuracy_cf], [x[1] for x in r_model_true_outcome_accuracy_cf], alpha=0.2)


    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Outcome Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../results/outcome_accuracy.png')
    plt.clf()


if __name__ == '__main__':
  create_dataset_from_folder('/Users/bigyankarki/Desktop/bigyan/results/async_jax/data/')


  

  
  

    
  



  
