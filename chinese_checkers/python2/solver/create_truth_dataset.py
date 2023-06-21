import os
import sys
import numpy as np
import itertools
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw

from mcts import mcts


solved_data = "/Users/bigyankarki/Desktop/bigyan/solved_data/"
cc = cw.CCheckers()
state = cw.CCState()
ranker = cw.CCLocalRank12()
solver = cw.Solver(solved_data, True, False)


def permute_states(board_slice_length, num_pieces):
    substate_strings = []
    substates = itertools.permutations(num_pieces * '2' + (board_slice_length - num_pieces)*'0')
    for substate in substates:
        if ''.join(substate) not in substate_strings:
            substate_strings.append(''.join(substate))
    return substate_strings


def remove_illegal(move_type, policy, state, cc, reverse=False):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)

    # for move in legal_moves:
    #     print(move.getFrom(), move.getTo())

    legals = pw.moves_existence(legal_moves, reverse)
    washed = legals * policy
    washed = washed / np.sum(washed)
    # washed = washed * tf.reshape((tf.ones([BOARD_SIZE, BOARD_SIZE]) - tf.eye(BOARD_SIZE)), [-1])
    # for move in legal_moves:
    #     cc.freeMove(move)

    return washed, legal_moves

def get_probabilities(washed, moves, reverse=False):
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


def get_label(state, solver):
    result = int(solver.lookup(state))
    if result == 1:
        label = -1.0
    elif result == 2:
        label = 1.0
    else:
        label = 0.0
    return label

def process_state_labels(cc, state, solver, move_type):

    # Generate list of moves for a state and
    # find the optimal ones.
    # Create a distribution over optimal actions.

    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)
    optimal_actions = len(legal_moves) * [0]
    labels = len(legal_moves) * [0]

    for (move_ind, move) in enumerate(legal_moves):
        cc.ApplyMove(state, move)
        label = get_label(state, solver)
        cc.UndoMove(state, move)
        labels[move_ind] = label

    if len(labels) > 0:
        max_label = max(labels)
    for move_ind in range(len(labels)):
        if labels[move_ind] == max_label:
            optimal_actions[move_ind] = 1
        else:
            optimal_actions[move_ind] = 0

    # print(labels)

    stats = []
    for move_ind, move in enumerate(legal_moves):
        stats.append(mcts.Node(move.getFrom(), move.getTo(), optimal_actions[move_ind], labels[move_ind], 0))

    # for move in legal_moves:
    #     move.Print(0)
    # print(optimal_actions)

    return int(get_label(state, solver)), stats


def rollout_terminal_states(cc, state, solver, move_type):
    pass


def process_terminal_states(cc, state, solver, move_type):
    get_moves = getattr(cc, 'get' + move_type)
    legal_moves = get_moves(state)
    legal_moves = pw.ll_to_list(legal_moves)
    optimal_actions = len(legal_moves) * [0]
    labels = len(legal_moves) * [0]

    for (move_ind, move) in enumerate(legal_moves):
        cc.ApplyMove(state, move)
        label = get_label(state, solver)
        if cc.Done(state):
            optimal_actions[move_ind] = 1
        cc.UndoMove(state, move)
        labels[move_ind] = label

    # assert(sum(optimal_actions) == 1)
    if sum(optimal_actions) == 0:
        print(state)

    stats = []
    for move_ind, move in enumerate(legal_moves):
        stats.append(mcts.Node(move.getFrom(), move.getTo(), optimal_actions[move_ind], labels[move_ind], 0))

    # for move in legal_moves:
    #     move.Print(0)
    # print(optimal_actions)

    return int(get_label(state, solver)), stats


# def create_terminal_states():
#     terminal_states = []

#     substate_strings = permute_states(cw.BOARD_SIZE**2 - 6, 3)
#     first_player_positions = ['100110', '100011', '010011', '010101', '001101', '001110']

#     for substate in substate_strings:
#         for position in first_player_positions:
#             terminal_states.append(substate + position)

#     substate_strings = permute_states(cw.BOARD_SIZE**2 - 6, 2)
#     first_player_positions = ['120110', '021110']

#     for substate in substate_strings:
#         for position in first_player_positions:
#             terminal_states.append(substate + position)

#     # print(terminal_states)

#     states = []
#     outcomes = []

#     for state_string in terminal_states:
#         cc.applyState(state_string, state)

#         if not cc.Done(state) and int(solver.lookup(state)) != 3:
#             label, stats = process_terminal_states(cc, state, solver, "MovesForward")

#             if state.getToMove():
#                 label = -label
#                 board = pw.reverse_state(state.getBoard())
#                 sample_dist = pw.moves_distribution(stats, cw.BOARD_SIZE, reverse=True)

#             else:
#                 board = state.getBoard()
#                 sample_dist = pw.moves_distribution(stats, cw.BOARD_SIZE)

#             if len(stats) != 0:
#                 outcomes.append(label)
#                 states.append([board, state.getToMove(), sample_dist])

#             log.create_tfrecord(states, outcomes,
#                                    '/home/zaheen/projects/sp_games/truth/4_4_3/terminals.tfrecord')
            
# function to create states from the solver
def create_all_states():
    state_info = {}

    for rank in range(ranker.getMaxRank()):
        ranker.unrank(rank, state)
        # state.PrintASCII()
        if not cc.Done(state) and int(solver.lookup(state)) != 3:
            label, stats = process_state_labels(cc, state, solver, "MovesForward")
            if state.getToMove():
                label = -label
                board = pw.reverse_state(state.getBoard())
                sample_dist = pw.moves_distribution(stats, cw.BOARD_SIZE, reverse=True)
            else:
                board = state.getBoard()
                sample_dist = pw.moves_distribution(stats, cw.BOARD_SIZE)
            if len(stats) != 0:
                print(rank)
                key = tuple(board)
                if key not in state_info:
                    state_info[tuple(board)] = [label, np.array(sample_dist)]

                if rank >= 1000000:
                    with open("state_info.pkl", "wb") as f:
                        pickle.dump(state_info, f)
                    return

                # np.savez("/Users/bigyankarki/Desktop/bigyan/5_5_6_test_data/rank_{}".format(rank), board=board, label=label, policy=sample_dist)

    with open("state_info.pkl", "wb") as f:
        pickle.dump(state_info, f)

    return

# # function to get true label from solver
# def get_label_from_solver(state):
#     label = solver.lookup(state)

#     if int(label) == 1:
#         label = -1
#     elif int(label) == 2:
#         label = 1
#     else:
#         label = 0
    
#     return label


# load the training states dictionary
def load_dict_of_states():
    states =  pickle.load(open('unique_training_states.p', 'rb'))
    return states

if __name__ == "__main__":
    trainings_states = load_dict_of_states()

    # measure the accuracy of the solver
    correct = 0
    total = 0

    for state, (player, policy, label) in trainings_states.items(): 
        # player = 0 if v[0] == 1 else 1
        cc_state = cw.list_to_ccstate(list(state), player) # convert the state to ccstate
        true_label, stats = process_state_labels(cc, cc_state, solver, "MovesForward")

        if state.getToMove():
            true_label = -true_label
            board = pw.reverse_state(state.getBoard())
            sample_dist = pw.moves_distribution(stats, cw.BOARD_SIZE, reverse=True)
        else:
            board = state.getBoard()
            sample_dist = pw.moves_distribution(stats, cw.BOARD_SIZE)

        if true_label == label:
            correct += 1
        total += 1



    print("Accuracy: {}".format(correct/total))