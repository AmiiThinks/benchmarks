# IMPALA Benchmark
Author: Subhojeet Pramanik

This repository contains a benchmark for training neural network models using Impala. IMPALA is a **reinforcement learning algorithm** that uses an **actor-critic** framework to learn from **experience trajectories**. It **decouples data collecting from learning** by having multiple actors that send trajectories of states, actions and rewards to a centralized learner. The learner then updates the policy and value networks using a GPU and mini-batches of trajectories. 

However, because the actors may use an outdated policy compared to the learner, the learning becomes **off-policy**. To correct for this discrepancy, IMPALA uses an **off-policy correction method** called **V-trace** that adjusts the importance weights and the target values of the trajectories.

IMPALA is designed for **scalable distributed deep reinforcement learning** and can handle a large collection of tasks using a single agent with a single set of parameters. It can achieve high throughput and data efficiency by parallelizing the actors and the learner.

## System Requirements

To run this benchmark, you will need:

- Docker: [Install Docker](https://www.docker.com/get-started)
- NVIDIA Docker support: [Install NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (required if using GPUs)

## Building the Docker Image

To build the Docker image, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the repository directory.
3. Open a terminal or command prompt.
4. Run the following command to build the Docker image:

```
docker build -t impala-benchmark .
```

This will build the Docker image with the necessary dependencies for running the benchmark.

## Running the Benchmark

To run the benchmark, use the following command:

```bash
sudo docker run impala-benchmark python3 train.py  --num_gpus=<NUM_GPUS>
```

Running the script should take around 10 min. At end the total execution time and average time for each training is printed. 


## Additional Options

The `train.py` script supports the following optional arguments:

- `-h`, `--help`: Show the help message and exit.
- `--num_gpus NUM_GPUS`: Specify the number of GPUs to use.


## Files

The project repository includes the following files:

- `Dockerfile`: Contains the instructions for building the Docker image.
- `README.md`: The file you are currently reading, providing instructions and information about the benchmark.
- `requirements.txt`: Lists the dependencies required for running the benchmark.
- `train.py`: The main script for training neural network models using Impala.

## Author
Subhojeet Pramanik (https://github.com/subho406)
