import os
import pickle
import argparse

def get_latest_models_parallel():
     # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--counter', type=int, default=0, help='No. of hours passed since the start of the experiment')
    args = parser.parse_args()
    counter = args.counter

    # get the latest nn_player from both single_play and parallel_play
    parallel_play_trained_path = "/Users/bigyankarki/Desktop/bigyan/results/async_jax/trained_models/"

    # find the latest mod and opt files
    try:
        parallel_play_latest_model_file = max([parallel_play_trained_path + f for f in os.listdir(parallel_play_trained_path) if f.startswith('mod')], key=os.path.getctime)
    except ValueError:
        print("No trained models found. Exiting...")
        return "Error"

    # save both of the files in a new folder
    latest_models_path = "/Users/bigyankarki/Desktop/bigyan/results/latest_models/"
    if not os.path.exists(latest_models_path):
        os.makedirs(latest_models_path)

    latest_models_path += "hour_" + str(counter) + "/"
    if not os.path.exists(latest_models_path):
        os.makedirs(latest_models_path)

    # get file numbers
    parallel_play_latest_model_file_number = parallel_play_latest_model_file.split("_")[-1].split(".")[0]
    parallel_play_file_name = "parallel_play_model_" + parallel_play_latest_model_file_number + ".pkl"

    
    # read the files and save them in the new folder
    with open(parallel_play_latest_model_file, 'rb') as fp:
        parallel_play_latest_model = pickle.load(fp)

    with open(os.path.join(latest_models_path, parallel_play_file_name), 'wb') as fp:
        pickle.dump(parallel_play_latest_model, fp)
    return latest_models_path

def get_latest_models_single():
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--counter', type=int, default=0, help='No. of hours passed since the start of the experiment')
    args = parser.parse_args()
    counter = args.counter

    # get the latest nn_player from both single_play and parallel_play
    single_play_trained_params_path = "/Users/bigyankarki/Desktop/bigyan/results/single_jax/trained_models/"

    # find the latest mod and opt files
    try:
        single_play_latest_model_file = max([single_play_trained_params_path + f for f in os.listdir(single_play_trained_params_path) if f.startswith('mod')], key=os.path.getctime)
    except ValueError:
        print("No trained models found. Exiting...")
        return "Error"

    # save both of the files in a new folder
    latest_models_path = "/Users/bigyankarki/Desktop/bigyan/results/latest_models/"
    if not os.path.exists(latest_models_path):
        os.makedirs(latest_models_path)

    latest_models_path += "hour_" + str(counter) + "/"
    if not os.path.exists(latest_models_path):
        os.makedirs(latest_models_path)

    # get file numbers
    single_play_latest_model_file_number = single_play_latest_model_file.split("_")[-1].split(".")[0]
    single_play_file_name = "single_play_model_" + single_play_latest_model_file_number + ".pkl"

    
    # read the files and save them in the new folder
    with open(single_play_latest_model_file, 'rb') as fp:
        single_play_latest_model = pickle.load(fp)
   
    with open(os.path.join(latest_models_path, single_play_file_name), 'wb') as fp:
        pickle.dump(single_play_latest_model, fp)
   
    return latest_models_path

def get_latest_models():
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--counter', type=int, default=0, help='No. of hours passed since the start of the experiment')
    args = parser.parse_args()
    counter = args.counter

    # get the latest nn_player from both single_play and parallel_play
    single_play_trained_params_path = "/Users/bigyankarki/Desktop/bigyan/results/single_jax/trained_models/"
    parallel_play_trained_path = "/Users/bigyankarki/Desktop/bigyan/results/async_jax/trained_models/"

    # find the latest mod and opt files
    try:
        single_play_latest_model_file = max([single_play_trained_params_path + f for f in os.listdir(single_play_trained_params_path) if f.startswith('mod')], key=os.path.getctime)
        parallel_play_latest_model_file = max([parallel_play_trained_path + f for f in os.listdir(parallel_play_trained_path) if f.startswith('mod')], key=os.path.getctime)
    except ValueError:
        print("No trained models found. Exiting...")
        return "Error"

    # save both of the files in a new folder
    latest_models_path = "/Users/bigyankarki/Desktop/bigyan/results/latest_models/"
    if not os.path.exists(latest_models_path):
        os.makedirs(latest_models_path)

    latest_models_path += "hour_" + str(counter) + "/"
    if not os.path.exists(latest_models_path):
        os.makedirs(latest_models_path)

    # get file numbers
    single_play_latest_model_file_number = single_play_latest_model_file.split("_")[-1].split(".")[0]
    parallel_play_latest_model_file_number = parallel_play_latest_model_file.split("_")[-1].split(".")[0]
    sinle_play_file_name = "single_play_model_" + single_play_latest_model_file_number + ".pkl"
    parallel_play_file_name = "parallel_play_model_" + parallel_play_latest_model_file_number + ".pkl"

    
    # read the files and save them in the new folder
    with open(single_play_latest_model_file, 'rb') as fp:
        single_play_latest_model = pickle.load(fp)
    with open(parallel_play_latest_model_file, 'rb') as fp:
        parallel_play_latest_model = pickle.load(fp)

    with open(os.path.join(latest_models_path, sinle_play_file_name), 'wb') as fp:
        pickle.dump(single_play_latest_model, fp)
    with open(os.path.join(latest_models_path, parallel_play_file_name), 'wb') as fp:
        pickle.dump(parallel_play_latest_model, fp)
    return latest_models_path

if __name__ == "__main__":
    get_latest_models_single()
