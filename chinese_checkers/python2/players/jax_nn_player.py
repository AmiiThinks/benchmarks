import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
from helpers import utils, log_helper
import os
import chex
from functools import partial


from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
import random as rnd
from models import AZeroModel

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

class NNPlayer():
    def __init__(self, config): 
        self._name = "nn_player"
        self.C = 1
        self.config = config
        self.trained_model_params_path = config["file"]["trained_model_params_path"]
        self.model = hk.transform_with_state(self._forward_fn)
        self.rng_key = jax.random.PRNGKey(42)
        self.dummy_x = jax.random.uniform(self.rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
        self.params, self.state = self.model.init(rng=self.rng_key, x=self.dummy_x)
        # self._load_trained_model(mod)

    
    def load_model(self, mod):
        if mod > 0:
            path = os.path.join(self.trained_model_params_path, 'mod_{}'.format(mod - 1))
            loaded_params = utils.load_model_params(path)
            self.params = loaded_params
        if mod > 0:
            chex.assert_trees_all_equal(self.params, loaded_params)
        return

    def _forward_fn(self, x):
        mod = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
        return mod(x)

    # def _load_trained_model(self, mod):
    #     if mod > 0:
    #         path = os.path.join(self.trained_model_params_path, 'mod_{}'.format(mod - 1))
    #         loaded_params = utils.load_model_params(path)
    #         self.params = loaded_params
    #     if mod > 0:
    #         chex.assert_trees_all_equal(self.params, loaded_params)
    #     return

    @partial(jax.jit, static_argnums=(0, 1))
    def _forward_pass (self, model, params, st, x, rng_key):
        forward, st = model.apply(params=params, state=st, x=x, rng=rng_key)
        y, policy = forward
        return y, policy
        
    def evaluate_state(self, state, cc):
        # return y
        if state.getToMove():
            input_rep = jnp.array(pw.reverse_state(state.getBoard()))
        else:
            input_rep = jnp.array(state.getBoard())
        
        p1 = pw.vec_to_board(jnp.array(jnp.expand_dims(input_rep, axis=0)).astype(np.int32), 1, 1)
        p2 = pw.vec_to_board(jnp.array(jnp.expand_dims(input_rep, axis=0)).astype(np.int32), 2, 1)
        inputs = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32)
    
        y, policy = self._forward_pass(self.model, params=self.params, st=self.state, x=inputs, rng_key=self.rng_key)

        return y[0][0]


    def expand_actions(self, cc, state, depth):
        if state.getToMove():
            input_rep = jnp.array(pw.reverse_state(state.getBoard()))
        else:
            input_rep = jnp.array(state.getBoard())

        p1 = pw.vec_to_board(jnp.array(jnp.expand_dims(input_rep, axis=0)).astype(np.int32), 1, 1)
        p2 = pw.vec_to_board(jnp.array(jnp.expand_dims(input_rep, axis=0)).astype(np.int32), 2, 1)
        inputs = jnp.array(jnp.stack((p1, p2), axis=3)).astype(np.float32) 

        # forward, st = self.model.apply(params=self.params, state=self.state, x=inputs, rng=self.rng_key)
       
        y, policy = self._forward_pass(self.model, params=self.params, st=self.state, x=inputs, rng_key=self.rng_key)
        # print("forward is: ", forward)
        # print(policy.keys())

        # policy = policy.numpy()[0]
        # remove all illegal moves and states here
        legals, legal_moves = self.remove_illegal('MovesForward', state, cc, state.getToMove())
        legals = jnp.array([legals])
        washed = jnp.where(jnp.array(legals).astype(np.bool), policy, -1e32*jnp.ones_like(policy))
        washed = jax.nn.softmax(washed)
        probs = self.get_probabilities(washed, legal_moves, state.getToMove())

        if depth == 0:
            num_legal_actions = len(probs)
            noise = np.random.dirichlet(num_legal_actions * [1/num_legal_actions], 1)[0] # noise
            for i in range(num_legal_actions):
                probs[i] = .75 * probs[i] + .25 * noise[i]
        return legal_moves, probs

    def select_action(self, root, depth):
        candidates = []
        max_value = -np.inf

        for child in root.children:
            value = child.value + (self.C * child.p * (np.sqrt(root.visit_count) / (1 + child.visit_count)))
            if value > max_value:
                candidates = [child]
                max_value = value
            elif value == max_value:
                candidates.append(child)
        return rnd.choice(candidates)

    def remove_illegal(self, move_type, state, cc, reverse=False):
        get_moves = getattr(cc, 'get' + move_type)
        legal_moves = get_moves(state)
        legal_moves = pw.ll_to_list(legal_moves)

        legals = pw.moves_existence(legal_moves, reverse)

        return legals, legal_moves

    def get_probabilities(self, washed, moves, reverse=False):
        total_size = (cw.BOARD_SIZE**2)
        washed = np.reshape(washed, [total_size, total_size])
        probs = []
        if reverse:
            for move in moves:
                p = washed[pw.mapping[total_size - 1 - move.getFrom()], pw.mapping[total_size - 1 - move.getTo()]]
                probs.append(p)
        else:
            for move in moves:
                p = washed[pw.mapping[move.getFrom()], pw.mapping[move.getTo()]]
                probs.append(p)

        return probs


if __name__ == "__main__":
    nn = NNPlayer()