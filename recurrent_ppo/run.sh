#!/bin/bash
docker run  -it recurrent_ppo python3 /root/recurrent_ppo/src/control_exps.py --config-name=config_ppo_lstm_popgym
docker run  -it recurrent_ppo python3 /root/recurrent_ppo/src/control_exps.py --config-name=config_ppo_gru_popgym
