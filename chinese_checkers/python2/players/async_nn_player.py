import numpy as np
import jax
import os
import jax.numpy as jnp
from helpers import utils, log_helper


from wrappers import pywrapper as pw
from wrappers import ccwrapper as cw
import random as rnd


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# get logger
logger = log_helper.get_logger("self_play")

class NNPlayer():

    def __init__(self, mod): 
        self._name = "nn_player"
        self.C = 1
        
    def evaluate_state(self, state, cc, y, policy):
        return y


    def expand_actions(self, cc, state, depth, y, policy):
        
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