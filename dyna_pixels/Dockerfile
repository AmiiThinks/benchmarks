FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget
# RUN apt-get install -y apt-transport-https curl gnupg
# RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
# RUN mv bazel.gpg /etc/apt/trusted.gpg.d/
# RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
# RUN apt-get update && apt-get install -y bazel
# RUN RUN bazel --version
# Install required Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# # Clone deepmind/lab repository and build pip package
# WORKDIR /tmp
# RUN git clone https://github.com/deepmind/lab.git && cd lab && \
#     bazel build -c opt --python_version=PY3 //python/pip_package:build_pip_package && \
#     ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg

# # Install deepmind/lab from the built package
# RUN pip install /tmp/dmlab_pkg/deepmind_lab-1.0-py2-none-any.whl --force-reinstall
# RUN pip3 install pandas tabulate scikit-image tqdm shimmy[gym] gymnasium[atari] gymnasium[accept-rom-license]
# RUN pip3 install gym[atari] autorom[accept-rom-license]
# Copy your train.py file
ADD src /app/

# Set working directory
WORKDIR /app

# Run the training script
CMD ["python3", "train.py", "--env_name=MiniGrid-Empty-8x8-v0-fullobs", \
     "--ae_model_type=soft_vqvae", "--filter_size=6", "--codebook_size=256", \
     "--trans_model_type=continuous", "--trans_depth=4", "--trans_hidden=512", \
     "--n_dyna_updates=4", "--n_trans_updates=4", "--dyna_batch_size=128", \
     "--dyna_steps=30_000", "--er_train_model", "--ae_recon_loss", "--device=cuda"]