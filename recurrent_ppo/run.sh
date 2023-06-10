#!/bin/bash
python recurrent_ppo/src/control_exps.py --config-name=config_ppo_lstm_popgym
python recurrent_ppo/src/control_exps.py --config-name=config_ppo_gru_popgym
