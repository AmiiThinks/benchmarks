# Build the container
`docker build -t <username>/<imagename>:<tag> .`

# Run the benchmark. Argument <world_size> to test.sh is the number of cores to use. Outputs written to stdout and pwd, in particular a file <world_size>-time.txt contains the time taken.
`docker run -v $(pwd):/bilevin/runs/docker -P <username>/<imagename>:<tag> /bin/bash -c "cd /bilevin && ./test.sh <world_size>"`
