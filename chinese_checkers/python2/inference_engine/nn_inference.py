import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
import os
import chex
import sys
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from helpers import utils, log_helper
from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
import random as rnd
from models import AZeroModel

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# get logger
logger = log_helper.get_logger("self_play_parallel")

# required game states
cc = cw.CCheckers()
gamestate = cw.CCState()
l = cw.CCLocalRank12()

# print the pid of the process
# print("infernce pid", os.getpid())

# filter out illegal moves
def remove_illegal(move_type, state, cc, reverse):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    # print(legal_moves)
    legal_moves = pw.ll_to_list(legal_moves)
    legals = pw.moves_existence(legal_moves, reverse)

    return legals

class NN_inference():
    def __init__(self, config): 
        self._name = "nn_inference"
        self.trained_model_params_path_parallel = config["file"]["trained_model_params_path"]
        self.model = hk.transform_with_state(self._forward_fn)
        self.rng_key = jax.random.PRNGKey(42)
        self.dummy_x = jax.random.uniform(self.rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
        self.params, self.st = self.model.init(rng=self.rng_key, x=self.dummy_x)


    # initialize model 
    def _forward_fn(self, x):
        mod = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
        return mod(x)

    def load_model(self, mod):
        if mod > 0:
            path = os.path.join(self.trained_model_params_path_parallel, 'mod_{}'.format(mod - 1))
            loaded_params = utils.load_model_params(path)
            self.params = loaded_params
         # assert that loaded params equal to params. (checking if params are loaded correctly)
        if mod > 0:
            chex.assert_trees_all_equal(self.params, loaded_params)
        return


    # def inference(self, state):
    #     input_rep = jnp.array(state)
    
    #     p1 = pw.vec_to_board(jnp.array(jnp.expand_dims(input_rep, axis=0)).astype(np.int32), 1, 1)
    #     p2 = pw.vec_to_board(jnp.array(jnp.expand_dims(input_rep, axis=0)).astype(np.int32), 2, 1)
    #     inputs = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32)
        
    #     forward, st = self.model.apply(params=self.params, state=self.state, x=inputs, rng=self.rng_key)
    #     y, policy = forward

    #     return y[0][0], policy
    
    @partial(jax.jit, static_argnums=(0,1))
    def _forward_pass (self, model, params, st, x, rng_key):
        forward, st = model.apply(params=params, state=st, x=x, rng=rng_key)
        y, policy = forward
        return y, policy

  
    def inference(self, states):
        batch_state = jnp.array(states)
        batch_size = len(states) # batch size

        # convert vector to board
        p1 = pw.vec_to_board(batch_state, 1, batch_size)
        p2 = pw.vec_to_board(batch_state, 2, batch_size)
        batch_state = jnp.array(jnp.stack((p1, p2), axis=3)).astype(jnp.float32) # stack the two boards together

        y, policy = self._forward_pass(self.model, self.params, self.st, batch_state, self.rng_key) # forward pass

        return y, policy

if __name__ == "__main__":
    state = [1,1,1,0,0,0,0,0,0,0,0,0,0,2,2,2]
    nn = NN_inference(0)
    nn.inference(state)


