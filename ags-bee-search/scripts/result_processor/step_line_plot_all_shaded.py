import subprocess
import sys
import statistics as st

import matplotlib.pylab as plt
import pandas as pd

import numpy as np
def get_time_in_seconds(timestr):
    time_splits = timestr.split(":")
    seconds = int(time_splits[0]) * 60 * 60 + int(time_splits[1]) * 60 + int(time_splits[2].split(".")[0])
    return seconds


models = ['1', '2', '3', '4', '5']

# Plot line styles for different models
line_styles = ['-', '--', '-.', ':', '-', '--']
max_line_style_index = len(line_styles) - 1
line_style_index = 0

if __name__ == "__main__":

    source_directory = sys.argv[1]
    isEasy = True
    
    # logs => all bustle like models -- [Buslte, BustleBFS, BustleBFS*] -- paths to the log files.
    logs = ["ags_bee_search.log", "bee_search.log"]
    logs2 = []
    logkeys = ["ags_bee_search", "bee_search"]
    logkeys2 = []  
    
    script_name = "./processor_script.sh"

    regenerate_csvs = True 

    if(regenerate_csvs):
        for model_number in models:
            for key, logfile in enumerate(logs):
                logkey = logkeys[key]
                print("LOG KEY: " + logkey)
                # sys.argv[1]/1_ags_bee_search.log
                # sys.argv[1]/1_bee_search.log
                script_result = subprocess.check_call([script_name, "./" + source_directory + "/" + model_number + "_" + logfile, logkey])
                print("script_result = " + str(script_result))

            benchmark_results = {}

            with open("bustle_benchmarks") as f:
                all_benchmarks = f.read().splitlines()

            for benchmark in all_benchmarks:
                stats = {"benchmark": benchmark}
                for logkey in logkeys:
                    stats[logkey + "_result"] = "Fail"
                    stats[logkey + "_program"] = "None"
                    stats[logkey + "_time"] = 3600
                    stats[logkey + "_evaluations"] = 100000000
                benchmark_results[benchmark] = stats

            for logkey in logkeys:
                print(logkey)
                with open(logkey + "_benchmarks.txt") as f:
                    benchmarks = f.read().splitlines()

                with open(logkey + "_results.txt") as f:
                    results = f.read().splitlines()

                with open(logkey + "_programs.txt") as f:
                    programs = f.read().splitlines()

                with open(logkey + "_evaluations.txt") as f:
                    evaluations = f.read().splitlines()

                with open(logkey + "_times.txt") as f:
                    times = f.read().splitlines()

                for index, benchmark in enumerate(benchmarks):

                    stats = benchmark_results.get(benchmark)
                    if stats is None:
                        stats = {"benchmark": benchmark}
                        benchmark_results[benchmark] = stats
                    stats[logkey + "_result"] = results[index]
                    stats[logkey + "_program"] = programs[index]
                    stats[logkey + "_time"] = get_time_in_seconds(str(times[index]))
                    stats[logkey + "_evaluations"] = evaluations[index]

            statistics = []

            for key, value in benchmark_results.items():
                statistics.append(value)

            dataFrame = pd.DataFrame(statistics)
            dataFrame.to_csv(source_directory + "/" + model_number + "-statistics.csv", index=False)


    combined_dataframe = pd.DataFrame()

    # 5 Models (1-5)
    # For each model, we have 3 different logkeys (bustle, bustle-bfs, bustle-bfs')
    # Get max evaluations. all 3, all 5 models. 
    # RESULT: Buslte-BFS', 3, xxxxx - list of evaluations 
    # FIX this list as x-axis.
    # For each approach (logkeys), for each model, I find the # of programs solved for a given evaluation.
    #  list of evaluations  = [200, 400, 500, ... MAX]
    #  model-2-bustle = [2,4,5,....179] 
    #  y = [1,1,5,6,7,89....,113]
    #  model-3-bustle = [1,3,3,....168] 
    #  avg => 1 List of # of programs solved for bustle.. 
    #  std => ... bustle 



    max_log_key = ""
    max_model_number = ""
    max_evals = 0
    max_time = 0
    max_time_model = ""
    max_time_log_key = ""
    for model_number in models:
        dataFrame = pd.read_csv(source_directory + "/" + model_number + "-statistics.csv")


        for lk in logkeys:

            # if result against current key is not "Success" then put time and evals to infinity

            # get the max evaluations
            if(dataFrame[lk + "_evaluations"].max() > max_evals):
                max_evals = dataFrame[lk + "_evaluations"].max()
                max_log_key = lk
                max_model_number = model_number

            # get the max time
            if(dataFrame[lk + "_time"].max() > max_time):
                max_time = dataFrame[lk + "_time"].max()
                max_time_model = model_number
                max_time_log_key = lk
            
            dataFrame.loc[dataFrame[lk + "_result"] != "Success", [lk + "_time", lk + "_evaluations"]] = float("inf"), float("inf")

        # Append column names with model number and append to combined_dataframe
        dataFrame.columns = [model_number + "-" + column for column in dataFrame.columns]
        combined_dataframe = pd.concat([combined_dataframe, dataFrame], axis=1)
    combined_dataframe.to_csv(source_directory + "/" + "combined-statistics.csv", index=False)


    # Print max evals data
    print("Max evaluations: " + str(max_evals))
    print("Max log key: " + max_log_key)
    print("Max model number: " + max_model_number)

    # Print max time data
    print("Max time: " + str(max_time))
    print("Max time model: " + max_time_model)
    print("Max time log key: " + max_time_log_key)

    # Get list of max evaluations
    max_evals_list = list(combined_dataframe[max_model_number + "-" + max_log_key + "_evaluations"])
    # Count no. of float infs in max_evals_list
    no_inf_evals = max_evals_list.count(float("inf"))
    # Remove infs from max_evals_list
    max_evals_list = [x for x in max_evals_list if x != float("inf")]
    # Sort the list
    max_evals_list.sort()
    # Get a list start from 1 to len(max_evals_list)
    # x_list = list(range(1, len(max_evals_list) + 1))

    evaluations_map = {}
    times_map = {}
    for index, log in enumerate(logs2):
        log_file = open(source_directory + "/" + log, 'r')
        log_file = log_file.readlines()
        temp_times = []
        temp_evals = []
        for line in log_file:
            time_taken = float(line.split("time-taken: ")[1].split(" ")[0])
            progs_evaluated = int(line.split("total-programs: ")[1].split(" ")[0])
            temp_times.append(time_taken)
            temp_evals.append(progs_evaluated)
        temp_times.sort()
        temp_evals.sort()
        times_map[logkeys2[index]] = temp_times
        evaluations_map[logkeys2[index]] = temp_evals


    # max_evaluations = max([max([max(evaluations_map[key]) for key in evaluations_map.keys()]), max_evals])
    # max_time_overall = max([max([max(times_map[key]) for key in times_map.keys()]), max_time])
    max_evaluations = max([max([max(evaluations_map[key]) for key in evaluations_map.keys()])] + [max_evals]) if evaluations_map else max_evals
    max_time_overall = max([max([max(times_map[key]) for key in times_map.keys()])] + [max_time]) if times_map else max_time


    print("Max evaluations overall: " + str(max_evaluations))
    print("Max time overall: " + str(max_time_overall))

    x_list = np.arange(1, max_evaluations + 1, 5e3)

    x_list_time = np.arange(1, max_time_overall + 1, 1)
    
    # For each value in max_evals_list, find the avg number of values <= that value against different models.
    avg_program_solved_against_evals = {}
    std_programs_solved_against_evals = {}

    for lk in logkeys:
        avg_program_solved_against_evals[lk] = []
        std_programs_solved_against_evals[lk] = []

        temp = {}
        for evals in x_list:
            # avg no. of programs solved agatins evals
            for model in models: 
                # if not model in temp: then initialize []
                if(not model in temp):
                    temp[model] = []
                evals_for_model = list(combined_dataframe[model + "-" + lk + "_evaluations"])
                # Count no. of values <= max_evals_list[i]
                temp[model].append(len([x for x in evals_for_model if x <= evals]))

        # Take mean of all the values in temp
        avg_prog_solved = np.array(list(temp.values())).transpose().mean(axis=1)
        avg_program_solved_against_evals[lk] = avg_prog_solved

        # Take std of all the values in temp
        std_prog_solved = np.array(list(temp.values())).transpose().std(axis=1)
        std_programs_solved_against_evals[lk] = std_prog_solved

    print(avg_program_solved_against_evals)
    print(std_programs_solved_against_evals)


    # Plot the data
    for lk in logkeys:
        # axis_fill
        plt.fill_between(x_list, avg_program_solved_against_evals[lk] - std_programs_solved_against_evals[lk], avg_program_solved_against_evals[lk] + std_programs_solved_against_evals[lk], alpha=0.2)
        
        # axis_line (with line_style)
        plt.step(x_list, avg_program_solved_against_evals[lk], label='BEE (82)' if lk == 'bee_search' else 'AGS + BEE (88)', linestyle=line_styles[line_style_index])
        line_style_index += 1

        # if(not isEasy):
        plt.text(x_list[-1], avg_program_solved_against_evals[lk][-1] - 2, round(avg_program_solved_against_evals[lk][-1]), fontsize=8.5)
        # else:
        #     if(lk == "BustleBFS"):
        #         # Set font size in plt.text
        #         plt.text(x_list[-1], avg_program_solved_against_evals[lk][-1] - 1, round(avg_program_solved_against_evals[lk][-1]), fontsize=8.5)
        #     elif(lk == "BustleBFS'"):
        #         plt.text(x_list[-1], avg_program_solved_against_evals[lk][-1] + 2, round(avg_program_solved_against_evals[lk][-1]), fontsize=8.5)
        #     elif(lk == "Bustle"):
        #         plt.text(x_list[-1], avg_program_solved_against_evals[lk][-1] - 6, round(avg_program_solved_against_evals[lk][-1]), fontsize=8.5)
                

    # max_solved = max([max([len(evaluations_map[key]) for key in evaluations_map.keys()]), len(max_evals_list)])
    max_solved = max([max([len(evaluations_map[key]) for key in evaluations_map.keys()])] + [len(max_evals_list)]) if evaluations_map else len(max_evals_list)

    for key, value in evaluations_map.items():
        total = len(value)
        expressions = list(value) 
        tasks = list(range(1, total + 1))
        plot_expressions = []
        plot_tasks = []
        if(total < max_solved):
            diff = max_solved - total
            for i in range(diff):
                expressions.append(max(expressions))
                tasks.append(total)

        # plt.step(plot_expressions, plot_tasks, label=key, where='pre')

        expressions.append(max_evaluations + 50000)
        tasks.append(tasks[-1] + 1)

        plot_expressions = expressions
        plot_tasks = tasks
        if key == "ags_bee_search":
            plt.plot(plot_expressions, plot_tasks, label='AGS + BEE (88)', linestyle=line_styles[line_style_index], line_width=2 if line_style_index > max_line_style_index else 1.5)
        else:
            plt.plot(plot_expressions, plot_tasks, label='BEE (82)', linestyle=line_styles[line_style_index], line_width=2 if line_style_index > max_line_style_index else 1.5)
        line_style_index += 1

        # if(not isEasy):
        plt.text(plot_expressions[-1] + 1, plot_tasks[-1] - 2, plot_tasks[-2], fontsize=8.5)
        # else:
        #     if(key == "BUS"):
        #         plt.text(plot_expressions[-1], plot_tasks[-1] - 4, plot_tasks[-2], fontsize=8.5)
        #     else:
        #         # set text font size to small.
        #         plt.text(plot_expressions[-1], plot_tasks[-1] - 2, plot_tasks[-2], fontsize=8.5)


    if(isEasy):
        # Limit y-axis [100, 205]
        plt.ylim(0, 90)
    else:
        # Limit y-axis [20, 140]
        plt.ylim(20, 140)

    # Increase font size of x, y ticks 
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)

    # Increase size of x, y Label Text.
    plt.xlabel("Number of Programs Evaluated", fontsize=14)
    plt.ylabel("Program synthesized", fontsize=14)

    #  Save fig as PDF.

    plt.title("Successes by expressions considered (SyGus)")

    plt.legend(loc='lower right', ncol=1, fontsize=14)
    plt.grid(True)
    plt.savefig("./result_images/" + source_directory + "evals.pdf", bbox_inches='tight')

    plt.show()
    plt.close()

    line_style_index = 0

    x_list_time = np.arange(1, max_time_overall + 1, 0.25)
    
    avg_program_solved_against_time = {}
    std_programs_solved_against_time = {}

    for lk in logkeys:
        avg_program_solved_against_time[lk] = []
        std_programs_solved_against_time[lk] = []

        temp = {}
        for current_time in x_list_time:
            # avg no. of programs solved agatins evals
            for model in models: 
                # if not model in temp: then initialize []
                if(not model in temp):
                    temp[model] = []
                time_for_model = list(combined_dataframe[model + "-" + lk + "_time"])
                # Count no. of values <= max_evals_list[i]
                temp[model].append(len([x for x in time_for_model if x <= current_time]))

        # Take mean of all the values in temp
        avg_prog_solved = np.array(list(temp.values())).transpose().mean(axis=1)
        avg_program_solved_against_time[lk] = avg_prog_solved

        # Take std of all the values in temp
        std_prog_solved = np.array(list(temp.values())).transpose().std(axis=1)
        std_programs_solved_against_time[lk] = std_prog_solved

    print(avg_program_solved_against_time)
    print(std_programs_solved_against_time)

    # Plot the data
    for lk in logkeys:
        # axis_fill
        plt.fill_between(x_list_time, avg_program_solved_against_time[lk] - std_programs_solved_against_time[lk], avg_program_solved_against_time[lk] + std_programs_solved_against_time[lk], alpha=0.2)
        plt.step(x_list_time, avg_program_solved_against_time[lk], label='BEE (82)' if lk == 'bee_search' else 'AGS + BEE (88)', linestyle=line_styles[line_style_index])
        line_style_index += 1

        # if(not isEasy):
        plt.text(x_list_time[-1], avg_program_solved_against_time[lk][-1] - 2, round(avg_program_solved_against_time[lk][-1]), fontsize=8.5)
        # else:
        #     if(lk == "BustleBFS"):
        #         plt.text(x_list_time[-1], avg_program_solved_against_time[lk][-1] - 1, round(avg_program_solved_against_time[lk][-1]), fontsize=8.5)
        #     elif(lk == "BustleBFS'"):
        #         plt.text(x_list_time[-1], avg_program_solved_against_time[lk][-1] + 2, round(avg_program_solved_against_time[lk][-1]), fontsize=8.5)
        #     elif(lk == "Bustle"):
        #         plt.text(x_list_time[-1], avg_program_solved_against_time[lk][-1] - 6, round(avg_program_solved_against_time[lk][-1]), fontsize=8.5)
        


    for key, value in times_map.items():
        total = len(value)
        times = list(value)
        tasks = list(range(1, total + 1))
        plot_times = []
        plot_tasks = []

        if(total < max_solved):
            diff = max_solved - total
            for i in range(diff):
                times.append(max(times))
                tasks.append(total)

        times.append(max_time_overall)
        tasks.append(tasks[-1] + 1)

        plot_times = times
        plot_tasks = tasks
        if key == "ags_bee_search":
            plt.plot(plot_times, plot_tasks, label='AGS + BEE (88)', where='pre', linestyle=line_styles[line_style_index], line_width=2 if line_style_index > max_line_style_index else 1.5)
        else:
            plt.plot(plot_times, plot_tasks, label='BEE (82)', where='pre', linestyle=line_styles[line_style_index], line_width=2 if line_style_index > max_line_style_index else 1.5)
        line_style_index += 1

        # if(not isEasy):
        plt.text(plot_times[-1] + 1, plot_tasks[-1] - 2, plot_tasks[-2], fontsize=8.5)
        # else:
        #     if(key == "BUS"):
        #         plt.text(plot_times[-1], plot_tasks[-1] - 4, plot_tasks[-2], fontsize=8.5)
        #     else:
        #         # set text font size to small.
        #         plt.text(plot_times[-1], plot_tasks[-1] - 2, plot_tasks[-2], fontsize=8.5)

    # for key,value in times_map.items():
    # times = list(value.keys())
    # tasks = list(value.values())
    # plt.step(times,tasks,label=key,where='pre')

    plt.xlabel("Time in seconds", fontsize=14)
    plt.ylabel("Program synthesized", fontsize=14)

    # Make legend with two columns
    plt.legend(loc='lower right', ncol=1, fontsize=14)

    if(isEasy):
        # Limit y-axis [100, 205]
        plt.ylim(0, 90)
    else:
        # Limit y-axis [20, 140]
        plt.ylim(20, 140)
    
    plt.title("Successes by time considered (SyGus)")
    plt.grid(True)
    plt.savefig("./result_images/" + source_directory + "time.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
