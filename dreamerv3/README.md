## Build Docker image

`docker build -t dreamerv3 .`

## Run image
`docker run --gpus all dreamerv3 `

## Run benchmark from interactive shell

`docker run --gpus all -it --entrypoint=/bin/bash dreamerv3`
`./benchmark.sh`

## logs located at

`app/dreamer_output.log `
