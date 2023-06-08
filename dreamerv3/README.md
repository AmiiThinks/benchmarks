## Build Docker image

`docker build -t dreamerv3 .`

## Run image

`docker run dreamerv3 `

## Run benchmark from interactive shell

`docker run -it --entrypoint=/bin/bash dreamerv3`
`./benchmark.sh`

## logs located at

`app/dreamer_output.log `
