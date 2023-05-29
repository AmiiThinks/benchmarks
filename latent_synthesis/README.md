# Benchmark for program VAE training and CEM on latent space execution

Author: Tales H. Carvalho (taleshen@ualberta.ca)

## Trainer Benchmark

Building Docker image:
```bash
docker build -t latentsyn_trainer_benchmark -f TrainerBenchmark/Dockerfile .
```

Executing benchmark (note: it might be necessary to include additional flags to allow GPU usage):
```bash
docker run latentsyn_trainer_benchmark
```

**Note**: The Docker image produced by Trainer Benchmark is very large (~10GB) because it is based on the official PyTorch image that contains CUDA and CudNN for GPU usage.

## CEM Benchmark

Building Docker image:
```bash
docker build -t latentsyn_cem_benchmark -f CEMBenchmark/Dockerfile .
```

Executing benchmark:
```bash
docker run latentsyn_cem_benchmark
```
