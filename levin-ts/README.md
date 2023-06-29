This benchmark uses a variable number of CPU core, and no GPU. **NOTE there is a bug in
this implementation that requires <world_size> to be greater than 1 (and <= the number of available cores).

Build the container:
```bash
docker build -t levin-ts .
```

Run the benchmark. Argument <world_size> to test.sh is the number of cores to use. Outputs written to stdout and a file <world_size>-time.txt contains the time taken:
```bash
docker run -v $(pwd):/bilevin/runs/docker -P levin-ts /bin/bash -c "cd /bilevin && ./test.sh <world_size>"
```
**Note**: the progress bar will not show incremental progress.
