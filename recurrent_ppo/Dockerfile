FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# install python3-pip
RUN apt update && apt install python3-pip -y

RUN pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install optax flax matplotlib wandb tqdm
RUN pip install hydra-core --upgrade

WORKDIR /usr/src/app 

COPY . .

RUN /bin/bash -c '/run.sh'