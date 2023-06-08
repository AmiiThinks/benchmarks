# IMPALA Benchmark
Author: Subhojeet Pramanik

This repository contains a benchmark for training neural network models using Impala. The benchmark can run different types of models.

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
sudo docker run impala-benchmark python3 train.py --model_type=<MODEL_TYPE> --num_gpus=<NUM_GPUS>
```

Replace `<MODEL_TYPE>` with the desired model type (`FF`, `LSTM`, or `TRF`) and `<NUM_GPUS>` with the number of GPUs to use.

For example, to train an LSTM model without using any GPUs, use the following command:

```bash
sudo docker run impala-benchmark python3 train.py --model_type=LSTM --num_gpus=0
```

Running the script should take around 10 min. At end the total execution time and average time for each training is printed. 

## Available Benchmarks

The benchmark supports the following model types:

- Feedforward Neural Network (FF)
- Long Short-Term Memory (LSTM)
- Transformer (TRF)

These models can be selected by specifying the corresponding `--model_type` option when running the benchmark.

## Additional Options

The `train.py` script supports the following optional arguments:

- `-h`, `--help`: Show the help message and exit.
- `--num_gpus NUM_GPUS`: Specify the number of GPUs to use.
- `--model_type MODEL_TYPE`: Specify the model type to use (`FF`, `LSTM`, or `TRF`).

Feel free to explore and modify the `train.py` script to suit your needs.

## Files

The project repository includes the following files:

- `Dockerfile`: Contains the instructions for building the Docker image.
- `README.md`: The file you are currently reading, providing instructions and information about the benchmark.
- `requirements.txt`: Lists the dependencies required for running the benchmark.
- `train.py`: The main script for training neural network models using Impala.

## Author
Subhojeet Pramanik (https://github.com/subho406)