# Build the container
`docker build -t <username>/<imagename>:<tag> .`


`docker run -v $(pwd):/bilevin/runs/docker -P <username>/<imagename>:<tag> /bin/bash -c "cd /bilevin && ./test.sh <world_size>"`
