You will need to have `nvidia-container-tools` installed to be able to use GPUs from within the docker container. See https://github.com/AmiiThinks/benchmarks/tree/main/latent_synthesis#setup-docker-to-use-gpu for details (for Arch based systems you can just install it from the AUR).

Build the container:
```bash
docker build -t crossbeam-grow .
```

Run the benchmark:
```bash
docker run --gpus all crossbeam-grow
```

**Note**: Depending on how how `nvidia-container-tools` is configured, you may need to add
`--runtime=nvidia` to the docker run command.
