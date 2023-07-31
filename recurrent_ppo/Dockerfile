FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
  
# install python3-pip
RUN apt update && apt install python3-pip -y

RUN pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install optax numpy==1.22 distrax rlax==0.1.5 flax==0.6.11 matplotlib wandb tqdm
RUN pip install hydra-core --upgrade
RUN pip install pandas gymnasium popgym

COPY . /root/recurrent_ppo/

WORKDIR /root/recurrent_pp
