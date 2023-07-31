#!/bin/bash
docker run --gpus all -it recurrent_ppo python3 /root/recurrent_ppo/src/control_exps.py --config-name=config_ppo_lstm_popgym
docker run --gpus all -it recurrent_ppo python3 /root/recurrent_ppo/src/control_exps.py --config-name=config_ppo_gru_popgym
