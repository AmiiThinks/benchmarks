# Benchmark for program VAE training and CEM on latent space execution

Author: Tales H. Carvalho (taleshen@ualberta.ca)

In this document, I list two options for running the benchmarks: Using Docker or Python's Virtualenv. In Compute Canada it is simpler to use Virtualenv as Docker is not natively supported, however using Singularity/Apptainer with the provided Dockerfile is also an available option.

## About the benchmarks

The two benchmarks in this folder are related to using neural program synthesis on a latent space to find programmatic policies that maximize reward in a Reinforcement Learning environment.

**TrainerBenchmark** tests the process of training a latent space of programs. This mainly relies on a GPU to train a neural model using PyTorch. The setup requires at least 32GB of GPU RAM to allocate the necessary memory. This benchmark is then evaluated by its completion time, available in the last line of stdout after the command execution.

**CEMBenchmark** tests the process of searching for a program in latent space. This mainly relies on multiple CPU processes to parallelize the search as much as possible. There is no constraint on the number of CPU cores, but the more the better. This benchmark is also evaluated by its completion time, available in the last line of stdout after the command execution.

## Environment setup

### Option 1: Setup Docker to use GPU

**Note**: This guide is based on [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Setup package repository and GPG key (assuming Ubuntu-based distribution):
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install `nvidia-container-toolkit`:
```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

Configure and restart Docker daemon:
```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

### Option 2: Setup Virtualenv

**Note**: Assuming python>=3.10.

This script creates a virtualenv and installs the necessary dependencies:
```bash
./create_venv.sh
```

## Executing the benchmarks

### TrainerBenchmark

#### Option 1: Using Docker

Building Docker image:
```bash
docker build -t latentsyn_trainer_benchmark -f TrainerBenchmark/Dockerfile .
```

Executing benchmark:
```bash
docker run --runtime=nvidia --gpus all latentsyn_trainer_benchmark
```

**Note 1**: In some systems, it is not necessary to include the flag `--runtime=nvidia`, but NVIDIA's documentation recommends it for Ubuntu-based systems.
**Note 2**: The Docker image produced by Trainer Benchmark is very large (~10GB) because it is based on the official PyTorch image that contains CUDA and CudNN for GPU usage.

#### Option 2: Using VirtualEnv

This script sources virtualenv and executes the benchmark:
```bash
./run_trainer_on_venv.sh
```

### CEMBenchmark

#### Option 1: Using Docker

Building Docker image:
```bash
docker build -t latentsyn_cem_benchmark -f CEMBenchmark/Dockerfile .
```

Executing benchmark:
```bash
docker run latentsyn_cem_benchmark
```

#### Option 2: Using Virtualenv

This script sources virtualenv and executes the benchmark:
```bash
./run_cem_on_venv.sh
```
