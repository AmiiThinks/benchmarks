import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from recurrent_ppo.src.utils import tree_index
from recurrent_ppo.src.model_fns.rnn import LSTMMultiLayer,GRUMultiLayer
from flax.linen.initializers import constant, orthogonal

def seq_model_lstm(**kwargs):
    def thurn():
        return LSTMMultiLayer(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'],reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return LSTMMultiLayer.initialize_state(kwargs['d_model'],kwargs['n_layers'])
    return thurn,initialize

def seq_model_gru(**kwargs):
    def thurn():
        return GRUMultiLayer(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'],
                             reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return GRUMultiLayer.initialize_state(kwargs['d_model'],kwargs['n_layers'])
    return thurn,initialize
