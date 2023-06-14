FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ADD src /app/

# Set working directory
WORKDIR /app

# Run the training script
CMD ["python3", "train.py", "--env_name=MiniGrid-Empty-8x8-v0-fullobs", \
     "--ae_model_type=soft_vqvae", "--filter_size=6", "--codebook_size=256", \
     "--trans_model_type=continuous", "--trans_depth=4", "--trans_hidden=512", \
     "--n_dyna_updates=4", "--n_trans_updates=4", "--dyna_batch_size=128", \
     "--dyna_steps=50_000", "--er_train_model", "--ae_recon_loss", "--device=cuda"]