```
 $ conda create --name rec_ppo --file requirements.txt
```

To run PPO+LSTM on POPGYM:
```
python recurrent_ppo/src/control_exps.py --config-name=config_ppo_lstm_popgym
```

To run PPO+RNN on POPGYM:
```
python recurrent_ppo/src/control_exps.py --config-name=config_ppo_gru_popgym
```
