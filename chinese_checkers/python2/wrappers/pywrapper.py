import sys
sys.path.append('../')
import numpy as np
# import tensorflow as tf
import copy
from wrappers import ccwrapper as cw
import jax.numpy as jnp
from jax import grad, jit, vmap




cc = cw.CCheckers()


class Node:
    def __init__(self, from_, to, numSamples):
        self.fromCell = from_
        self.toCell = to
        self.numSamples = numSamples


def ll_to_list(moves_ll):
    """
    Converts a linked list of CCMove objects to a list of CCMove objects. Helper to make
    working with a number of moves easier in Python.

    :param moves_ll: linked list containing moves
    :return: a list of CCMove objects
    """

    moves = []

    move = moves_ll

    while move is not None:
        moves.append(move)
        tmp = move.getNextMove()
        move.setNextMove(None)
        cc.freeMove(move)
        move = tmp

    return moves


def moves_distribution(nodes, dim, reverse=False, turn=0):
    num_spots = dim ** 2
    distribution = (num_spots) ** 2 * [0]
    total_samples = 0

    for move in nodes:
        if reverse:
            to_cell = (num_spots - 1) - move.toCell
            from_cell = (num_spots - 1) - move.fromCell
        else:
            to_cell = move.toCell
            from_cell = move.fromCell

        distribution[mapping[from_cell] * num_spots + mapping[to_cell]] = move.numSamples
        total_samples += move.numSamples
    
    if turn > 10:
        tau = 0.1
    else:
        tau = 1
    distribution = np.power(distribution, (1 / tau))
    if np.sum(distribution) != 0:
        distribution /= np.sum(distribution)
    
    return distribution


def moves_existence(moves, reverse=False):
    num_spots = cw.BOARD_SIZE ** 2
    distribution = num_spots ** 2 * [0]

    for move in moves:
        if reverse:
            to_cell = (num_spots - 1) - move.getTo()
            from_cell = (num_spots - 1) - move.getFrom()
        else:
            to_cell = move.getTo()
            from_cell = move.getFrom()

        distribution[mapping[from_cell] * num_spots + mapping[to_cell]] = 1

    return distribution


def board_mapping(dim):
    mapping = dim * dim * [0]
    turn = 1
    base_ind = 0
    for i in range(dim):
        ind = base_ind
        for j in range(dim - 1):
            mapping[ind] = i + j * dim

            if turn + j >= dim:
                offset = dim - (turn + j - dim + 1)
            else:
                offset = turn + j

            ind += offset
        mapping[ind] = i + (j + 1) * dim

        turn += 1
        base_ind += turn

    return np.array(mapping)


def reverse_state(board):
    reverse_board = copy.deepcopy(board)
    for i in range(len(reverse_board)):
        if reverse_board[i] == 1:
            reverse_board[i] = 2
        elif reverse_board[i] == 2:
            reverse_board[i] = 1

    return np.flip(reverse_board)

def tf_board_mapping(dim):
    mapping = dim * dim * [0]
    turn = 1
    base_ind = 0
    for i in range(dim):
        ind = base_ind
        for j in range(dim - 1):
            mapping[ind] = i + j * dim

            if turn + j >= dim:
                offset = dim - (turn + j - dim + 1)
            else:
                offset = turn + j

            ind += offset
        mapping[ind] = i + (j + 1) * dim

        turn += 1
        base_ind += turn
    return tf.convert_to_tensor(mapping, dtype=tf.int64)


def unbatched_gather_nd(params, indices):
    return params[tuple(jnp.moveaxis(indices, -1, 0))]

def batched_gather_nd(params, indices):
    return vmap(unbatched_gather_nd, (None, 0), 0)(params, indices)

def vec_to_board(vector, player, batch_size):
    board = np.tile(jnp.expand_dims(jnp.array(range(cw.BOARD_SIZE**2)), 0), [batch_size, 1])
    tiled_mapping = jnp.tile(jnp.expand_dims(tf_mapping, 0), [batch_size, 1])
    player_pos = jnp.where(vector == player, None, None)
    player_pos_stacked = jnp.stack(player_pos, axis=-1)
    locations = jnp.expand_dims(jnp.reshape(batched_gather_nd(tiled_mapping, player_pos_stacked), [batch_size, -1]), axis=2)
    board = jnp.any(jnp.subtract(jnp.expand_dims(board, axis=1), locations) == 0, axis=1)
    return jnp.reshape(board, [batch_size, cw.BOARD_SIZE, cw.BOARD_SIZE])



mapping = board_mapping(cw.BOARD_SIZE)
reverse_mapping = ((cw.BOARD_SIZE ** 2) - 1) - mapping
tf_mapping = board_mapping(cw.BOARD_SIZE)


# if __name__ == "__main__":
#     vector = jnp.array([[1,1,1,0,0,0,0,0,0,0,0,0,0,2,2,2]])
#     player = 1
#     batch_size = 1
#     vmap(vec_to_board, (0,0), 0)(vector, player, batch_size)

# if __name__ == "__main__":
    # vector = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1] + 14 * [0] + [1]
    # reverse_vector = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1] + 14 * [0] + [1]
    # print(vector)
    # reverse_vector.reverse()
    # print(vector)
    # # print(reverse_state(vector))
    # # tf_vec_to_board(tf.constant(vector, dtype=tf.int64), 1, 2)
    # # tf_vec_to_board(vector, 2, 1)
    # # print(mapping)
    # # print(reverse_mapping)
    # print(vec_to_board(np.array(vector), 1, 5))
    # print(vec_to_board(np.array(reverse_vector), 1, 5))
    # policy_to_moves(None)



