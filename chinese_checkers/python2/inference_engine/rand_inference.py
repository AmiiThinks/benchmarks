import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
from helpers import utils, log_helper
import json
import os
import chex


from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
import random as rnd
from models import AZeroModel

 # get hyperparameters and constants
with open('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/config/config.json') as f:
    config = json.load(f)

trained_model_params_path = config["file"]["trained_model_params_path"]

# get logger
logger = log_helper.get_logger("self_play")

class RAND_inference():
    
    def __init__(self): 
        self._name = "rand_player"
        
    def inference(self, state):
        # get either -1 or 1 randomly
        y = []
        for i in range(len(state)):
            y.append(rnd.choice([-1, 1]))

        # get random number between 0 and 1, of size 256
        policy = [np.random.rand(256)] * len(state)

        return y, policy