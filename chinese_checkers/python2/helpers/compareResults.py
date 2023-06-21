# file to compare the results of two different models

import os
import pickle
import matplotlib.pyplot as plt


def plot_iteration_comparions():
    async_jax_path = "/Users/bigyankarki/Desktop/bigyan/results/39_res_blocks/async_jax_plots/iteration_stats.pkl"
    single_jax_path = "/Users/bigyankarki/Desktop/bigyan/results/39_res_blocks/single_jax_plots/iteration_stats.pkl"
    # jax_path = "/Users/bigyankarki/Desktop/bigyan/results/jax/parallel/plots/iteration_stats.pkl"

    async_jax_pickle = {}
    jax_pickle = {}
    with open(async_jax_path, 'rb') as fp:
        async_jax_pickle = pickle.load(fp)
    with open(single_jax_path, 'rb') as fp:
        jax_pickle = pickle.load(fp)


    # plot
    single_play_iterations = []
    parallel_play_iterations = []
    async_jax_average_times = []
    async_jax_game_counts = []
    async_jax_average_states = []
    async_jax_player_1_wins = []
    async_jax_player_2_wins = []
    async_jax_draws = []
    async_jax_training_times = []

    jax_average_times = []
    jax_game_counts = []
    jax_average_states = []
    jax_player_1_wins = []
    jax_player_2_wins = []
    jax_draws = []
    jax_training_times = []
    
    for iteration in list(async_jax_pickle.keys())[:-1]:
        # if iteration > 52:
        #     break
        parallel_play_iterations.append(iteration)
        async_jax_average_times.append(async_jax_pickle[iteration]["average_time"])
        async_jax_game_counts.append(async_jax_pickle[iteration]["game_count"])
        async_jax_average_states.append(async_jax_pickle[iteration]["average_states"])
        async_jax_player_1_wins.append(async_jax_pickle[iteration]["player_win_status"][1])
        async_jax_player_2_wins.append(async_jax_pickle[iteration]["player_win_status"][2])
        async_jax_draws.append(async_jax_pickle[iteration]["player_win_status"][0])
        async_jax_training_times.append(async_jax_pickle[iteration]["training_time"])

    
    for iteration in list(jax_pickle.keys())[:-1]:
        single_play_iterations.append(iteration)
        jax_average_times.append(jax_pickle[iteration]["average_time"])
        jax_game_counts.append(jax_pickle[iteration]["game_count"])
        jax_average_states.append(jax_pickle[iteration]["average_states"])
        jax_player_1_wins.append(jax_pickle[iteration]["player_win_status"][1])
        jax_player_2_wins.append(jax_pickle[iteration]["player_win_status"][2])
        jax_draws.append(jax_pickle[iteration]["player_win_status"][0])
        jax_training_times.append(jax_pickle[iteration]["training_time"])

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax[0, 0].plot(parallel_play_iterations, async_jax_average_times, label="Parallel play")
    ax[0, 0].plot(single_play_iterations, jax_average_times, label="Single play")
    ax[0, 0].set_title("Average Time to play a game")
    ax[0, 0].set_xlabel("Iteration")
    ax[0, 0].set_ylabel("Average Time (s)")
    ax[0, 0].legend()

    ax[0, 1].plot(parallel_play_iterations, async_jax_game_counts, label="Parallel play")
    ax[0, 1].plot(single_play_iterations, jax_game_counts, label="Single play")
    ax[0, 1].set_title("No. of games played")
    ax[0, 1].set_xlabel("Iteration")
    ax[0, 1].set_ylabel("Game Count")
    ax[0, 1].legend()

    ax[1, 0].plot(parallel_play_iterations, async_jax_average_states, label="Parallel play")
    ax[1, 0].plot(single_play_iterations, jax_average_states, label="Single play")
    ax[1, 0].set_title("Average States visited in a game")
    ax[1, 0].set_xlabel("Iteration")
    ax[1, 0].set_ylabel("Average States")
    ax[1, 0].legend()

    ax[1, 1].plot(parallel_play_iterations, async_jax_player_1_wins, label="p1 wins (parallel play)")
    ax[1, 1].plot(parallel_play_iterations, async_jax_player_2_wins, label="P2 wins (parallel play)")
    # ax[1, 1].plot(parallel_play_iterations, async_jax_draws, label="Draws (parallel play)")
    ax[1, 1].plot(single_play_iterations, jax_player_1_wins, label="p1 wins (single play)")
    ax[1, 1].plot(single_play_iterations, jax_player_2_wins, label="P2 wins (single play)")
    # ax[1, 1].plot(single_play_iterations, jax_draws, label="Draws (single play)")
    ax[1, 1].set_title("Player Win Status")
    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("Player Win Status")
    ax[1, 1].legend()

    path = os.path.join('../../results/', "comparison.png")
    plt.savefig(path)
    plt.close()

    # plot training times
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(parallel_play_iterations, async_jax_training_times, label="Parallel play")
    ax.plot(single_play_iterations, jax_training_times, label="Single play")
    ax.set_title("Training Time")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Time (s)")
    ax.legend()

    # print(os.getpwd())
    path = os.path.join('../../results/', "training_time_comparison.png")
    plt.savefig(path)
    plt.close()

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


def plot_training_comparisons():
    async_jax_path = "/Users/bigyankarki/Desktop/bigyan/results/39_res_blocks/async_jax_plots"
    jax_path = "/Users/bigyankarki/Desktop/bigyan/results/39_res_blocks/single_jax_plots"

    async_jax_total_loss_path = os.path.join(async_jax_path, "total_loss.pkl")
    async_jax_policy_loss_path = os.path.join(async_jax_path, "policy_loss.pkl")
    async_jax_value_loss_path = os.path.join(async_jax_path, "value_loss.pkl")

    jax_total_loss_path = os.path.join(jax_path, "total_loss.pkl")
    jax_policy_loss_path = os.path.join(jax_path, "policy_loss.pkl")
    jax_value_loss_path = os.path.join(jax_path, "value_loss.pkl")

    async_jax_iterations = []
    async_jax_total_losses = []
    async_jax_policy_losses = []
    async_value_losses = []

    jax_iterations = []
    jax_total_losses = []
    jax_policy_losses = []
    jax_value_losses = []

    with open(async_jax_total_loss_path, "rb") as f:
        async_jax_total_losses, async_jax_iterations = _update_losses(pickle.load(f))

    with open(async_jax_policy_loss_path, "rb") as f:
        async_jax_policy_losses, _ = _update_losses(pickle.load(f))

    with open(async_jax_value_loss_path, "rb") as f:
        async_value_losses, _ = _update_losses(pickle.load(f))

    with open(jax_total_loss_path, "rb") as f:
        jax_total_losses, jax_iterations = _update_losses(pickle.load(f))

    with open(jax_policy_loss_path, "rb") as f:
        jax_policy_losses, _ = _update_losses(pickle.load(f))

    with open(jax_value_loss_path, "rb") as f:
        jax_value_losses,_ = _update_losses(pickle.load(f))

    # plot the losses
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(async_jax_iterations, async_jax_total_losses, label="Parallel play")
    ax.plot(jax_iterations, jax_total_losses, label="Single play")
    ax.set_title("Total Loss")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Total Loss")
    ax.legend()
    path = os.path.join('../../results/', "total_loss_comparison.png")
    plt.savefig(path)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(async_jax_iterations, async_jax_policy_losses, label="Parallel play")
    ax.plot(jax_iterations, jax_policy_losses, label="Single play")
    ax.set_title("Policy Loss")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Policy Loss")
    ax.legend()
    path = os.path.join('../../results/', "policy_loss_comparison.png")
    plt.savefig(path)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(async_jax_iterations, async_value_losses, label="Parallel play")
    ax.plot(jax_iterations, jax_value_losses, label="Single play")
    ax.set_title("Value Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value Loss")
    ax.legend()
    path = os.path.join('../../results/', "value_loss_comparison.png")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    plot_iteration_comparions()
    plot_training_comparisons()




        