from matplotlib import pyplot as plt
import os
import pickle


# class to help keep track of iteration statistics
class IterationStats:
    def __init__(self, config):
        self.iteration_stats = {}
        self.training_time = 0
        self.config = config

    def update(self, iteration, total_time, game_count, states, winner):
        if iteration in self.iteration_stats.keys():
            self.iteration_stats[iteration]["average_time"] = (self.iteration_stats[iteration]["average_time"] + total_time)/2
            self.iteration_stats[iteration]["game_count"] = game_count
            self.iteration_stats[iteration]["average_states"] = (self.iteration_stats[iteration]["average_states"] + len(states))/2
            self.iteration_stats[iteration]["player_win_status"][winner] += 1
        else:
            self.iteration_stats[iteration] = {"average_time": total_time, "game_count": game_count, "average_states": len(states), "player_win_status": {0:0, 1:0, 2:0}, "training_time": 0}
            self.iteration_stats[iteration]["player_win_status"][winner] = 1

    def update_training_time(self, iteration, training_time):
        if iteration in self.iteration_stats.keys():
            self.iteration_stats[iteration]["training_time"] = training_time
            # self.save_stats()

    # save stats as pickle file
    def save_stats(self):
        with open(os.path.join(self.config["file"]["training_plots_path"], "iteration_stats.pkl"), "wb") as f:
            pickle.dump(self.iteration_stats, f)
        # logger.info("Saved iteration stats to {}".format(os.path.join(self.config["training_plots_path"], "iteration_stats.pkl")))

    def print_stats(self, logger):
        logger.info(" Iteration | Average Time | Game Count | Average States | Draw | Player 1 Win | Player 2 Win | Training Time")
        for iteration in self.iteration_stats.keys():
            logger.info(" {:^12} | {:^12.2f} | {:^12} | {:^12.2f} | {:^12} | {:^12} | {:^12} | {:^12.2f}".format(iteration, self.iteration_stats[iteration]["average_time"], self.iteration_stats[iteration]["game_count"], self.iteration_stats[iteration]["average_states"], self.iteration_stats[iteration]["player_win_status"][0], self.iteration_stats[iteration]["player_win_status"][1], self.iteration_stats[iteration]["player_win_status"][2]), self.iteration_stats[iteration]["training_time"])
            
    def get_stats(self):
        return self.iteration_stats
    
    def log_stats(self, iteration, logger):
        print_list = [iteration, self.iteration_stats[iteration]["average_time"], self.iteration_stats[iteration]["game_count"], self.iteration_stats[iteration]["average_states"], self.iteration_stats[iteration]["player_win_status"][0], self.iteration_stats[iteration]["player_win_status"][1], self.iteration_stats[iteration]["player_win_status"][2], self.iteration_stats[iteration]["training_time"]]
        logger.info(" {:^12} | {:^12.2f} | {:^12} | {:^12.2f} | {:^12} | {:^12} | {:^12} | {:^12f}".format(*print_list))

    def plot_stats(self):
        iterations = []
        average_times = []
        game_counts = []
        average_states = []
        player_1_wins = []
        player_2_wins = []
        draws = []
        training_times = []
        
        for iteration in list(self.iteration_stats.keys())[:-1]:
            iterations.append(iteration)
            average_times.append(self.iteration_stats[iteration]["average_time"])
            game_counts.append(self.iteration_stats[iteration]["game_count"])
            average_states.append(self.iteration_stats[iteration]["average_states"])
            player_1_wins.append(self.iteration_stats[iteration]["player_win_status"][1])
            player_2_wins.append(self.iteration_stats[iteration]["player_win_status"][2])
            draws.append(self.iteration_stats[iteration]["player_win_status"][0])
            training_times.append(self.iteration_stats[iteration]["training_time"])

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
        ax[1, 0].set_title("Average no. states visited")
        ax[1, 0].set_xlabel("Iteration")
        ax[1, 0].set_ylabel("Average States")

        ax[1, 1].plot(iterations, player_1_wins, label="Player 1 Wins")
        ax[1, 1].plot(iterations, player_2_wins, label="Player 2 Wins")
        ax[1, 1].plot(iterations, draws, label="Draws")
        ax[1, 1].set_title("Player Win Status")
        ax[1, 1].set_xlabel("Iteration")
        ax[1, 1].set_ylabel("Win Status")
        ax[1, 1].legend()

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(iterations, training_times)
        ax.set_title("Training Time")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training Time (s)")
        path = os.path.join(self.config["file"]["training_plots_parallel"], "stats.png")
        plt.savefig(path)
        plt.close()






