The purpose of this benchmark is to time dreamerv3 running in different environments. The logs located at /dreamer/dreamer_output.log contain output from dreamerv3 as well as output from the linux time commands for each of three environments: dm_control, crafter, and atari. Please collect and return the log file. 

## Build Docker image

`docker build -t dreamerv3 .`

## Run image
Running the script should take around 15 min. At the end, the log file will be printed to the screen. It contains the execution time for each of the 3 environments.
`docker run --gpus all dreamerv3 `

## Run benchmark from interactive shell

`docker run --gpus all -it --entrypoint=/bin/bash dreamerv3`
`./benchmark.sh`

## logs located at

`/dreamer/dreamer_output.log `


## Example time for each env
![Screenshot from 2023-06-20 15-55-22](https://github.com/AmiiThinks/benchmarks/assets/48653578/ab642444-b8da-406f-b9eb-84e9aed6e797)
![Screenshot from 2023-06-20 15-55-08](https://github.com/AmiiThinks/benchmarks/assets/48653578/13694209-94e6-42e0-a3e1-8d6821018c5b)
![Screenshot from 2023-06-20 15-54-53](https://github.com/AmiiThinks/benchmarks/assets/48653578/625ca187-8f40-4b4c-9ffa-ffb23a624a2c)
