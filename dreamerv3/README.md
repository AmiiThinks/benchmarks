The purpose of this benchmark is to time dreamerv3 running in different environments. The logs located at /dreamer/dreamer_output.log contain output from dreamerv3 as well as output from the linux time commands for each of three environments: dm_control, crafter, and atari. Please collect and return the log file. 

## Build Docker image

`docker build -t dreamerv3 .`

## Run image
`docker run --gpus all dreamerv3 `

## Run benchmark from interactive shell

`docker run --gpus all -it --entrypoint=/bin/bash dreamerv3`
`./benchmark.sh`

## logs located at

`/dreamer/dreamer_output.log `
