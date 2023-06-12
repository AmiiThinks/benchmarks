FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
  
# install python3-pip
RUN apt update && apt install python3-pip -y

RUN pip install "jax[gpu]"
RUN pip install optax numpy==1.22 distrax rlax==0.1.5 flax matplotlib wandb tqdm
RUN pip install hydra-core --upgrade
RUN pip install pandas gymnasium popgym

COPY . /root/recurrent_ppo/

WORKDIR /root/recurrent_pp