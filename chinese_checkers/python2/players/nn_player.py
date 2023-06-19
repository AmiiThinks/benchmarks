import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp
from helpers import utils, log_helper
import json
import os
import chex
from functools import partial


from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
import random as rnd
from models import AZeroModel

 # get hyperparameters and constants
with open('/Users/bigyankarki/Desktop/bigyan/cc/chinese-checkers/python2/config/config.json') as f:
    config = json.load(f)

trained_model_params_path = config["file"]["trained_model_params_path"]
trained_model_params_path_parallel = config["file"]["trained_model_params_path_parallel"]

# get logger
logger = log_helper.get_logger("self_play")


class NNPlayer():
    # initialize model 
    def _forward_fn(self, x):
        mod = AZeroModel.AlphaZeroModel(num_filters=256, training=True)
        return mod(x)
 
    
    def __init__(self, mod): 
        self._name = "nn_player"
        self.C = 1
        self.model = hk.transform_with_state(self._forward_fn)
        self.rng_key = jax.random.PRNGKey(42)
        self.dummy_x = jax.random.uniform(self.rng_key, (1, cw.BOARD_SIZE, cw.BOARD_SIZE, 2))
        self.params, self.state = self.model.init(rng=self.rng_key, x=self.dummy_x)
        self._load_trained_model(mod)

    def _load_trained_model(self, mod):
        if mod > 0:
            logger.info("Loading model from: mod_{}".format(mod-1))
            # path = os.path.join(trained_model_params_path, 'mod_{}'.format(mod - 1))
            path = os.path.join(trained_model_params_path_parallel, 'mod_{}'.format(mod - 1))

            loaded_params = utils.load_model_params(path)
            self.params = loaded_params
        
         # assert that loaded params equal to params. (checking if params are loaded correctly)
        if mod > 0:
            chex.assert_trees_all_equal(self.params, loaded_params)

        return

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
        
        # forward, st = self.model.apply(params=self.params, state=self.state, x=inputs, rng=self.rng_key)
    
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
        # moves, probs = self.policy_to_moves(washed, state.getToMove())
        probs = self.get_probabilities(washed, legal_moves, state.getToMove())

        # moves = []
        # for move in legal_moves:

        # for ind, move in enumerate(moves):
        #     print(move.getFrom(), move.getTo(), probs[ind])
        if depth == 0:
            num_legal_actions = len(probs)
            noise = np.random.dirichlet(num_legal_actions * [1/num_legal_actions], 1)[0] # noise
            for i in range(num_legal_actions):
                probs[i] = .75 * probs[i] + .25 * noise[i]
        
        #print(sum(probs))

        return legal_moves, probs

    def select_action(self, root, depth):
        candidates = []
        max_value = -np.inf

        for child in root.children:
            value = child.value + (self.C * child.p * (np.sqrt(root.visit_count) / (1 + child.visit_count)))
            # print(value, child.move.getFrom(), child.move.getTo())
            if value > max_value:
                candidates = [child]
                max_value = value
            elif value == max_value:
                candidates.append(child)
        # print(len(candidates))
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
        # projection = np.argsort(pw.mapping)
        probs = []
        if reverse:
            for move in moves:
                # print(move.getFrom(), move.getTo())
                p = washed[pw.mapping[total_size - 1 - move.getFrom()], pw.mapping[total_size - 1 - move.getTo()]]
                probs.append(p)
        else:
            for move in moves:
                # print(move.getFrom(), move.getTo())
                p = washed[pw.mapping[move.getFrom()], pw.mapping[move.getTo()]]
                probs.append(p)

        return probs


if __name__ == "__main__":
    nn = NNPlayer()

    # np.random.seed()
    # node_list = [pw.Node(2, 13, 2),
    #              pw.Node(1, 6, 10),
    #              pw.Node(5, 9, 10)]

    # cc = cw.CCheckers()
    # state = cw.CCState()
    # cc.Reset(state)

    # print(state.getBoard())
    # # fm = cc.getMovesForward(state)
    # # fm = pw.ll_to_list(fm)
    # move = cc.getNewMove()
    # move.from_ = 3
    # move.to_ = 6
    # # move = fm[1]
    # move.setNextMove(None)
    # action = move.clone(cc)
    # # cc.freeMove(move)
    # #cc.ApplyMove(state, action)
    # # cc.freeMove(action)
    # print(state.getBoard())
    # move = cc.getNewMove()
    # move.from_ = 21
    # move.to_ = 18
    # move.setNextMove(None)
    # action = move.clone(cc)
    # # cc.freeMove(move)
    # #cc.ApplyMove(state, action)
    # # cc.freeMove(action)
    # print(state.getBoard())
    # # policy = np.random.random(625)
    # # print(policy)
    # nn_player = NNPlayer(None)

    # # nn_player.expand_actions(cc, state)

    # policy = np.array(pw.moves_distribution(node_list, 5))

    # movesForward = cc.getMovesForward(state)
    # movesForward = pw.ll_to_list(movesForward)
    # for m in movesForward:
    #     print(m.getFrom(), m.getTo())
    # #
    # print(policy)
    # washed, legal_moves = nn_player.remove_illegal('MovesForward', policy, state, cc, state.getToMove())
    # probs = nn_player.get_probabilities(washed, legal_moves, state.getToMove())
    # print(probs)
    # num_legal_actions = len(probs)
    # noise = np.random.dirichlet(num_legal_actions * [1/num_legal_actions], 1)[0]
    # print(noise)
    # for i in range(num_legal_actions):
    #             probs[i] = .9 * probs[i] + .1 * noise[i]
    # print(probs)