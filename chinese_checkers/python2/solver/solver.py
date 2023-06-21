
import sys
import random
from multiprocessing import Pool, set_start_method
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from wrappers import ccwrapper as cw
from wrappers import pywrapper as pw
from players import eval_nn_player as nnp
from mcts import mcts

from players import nn_player as nnp, random_player as rp, uct_player as uct


cc = cw.CCheckers()
game_state = cw.CCState()

solved_data = '/Users/bigyankarki/Desktop/bigyan/cc/results/solver'
solver = cw.Solver(solved_data, True, False)


# def get_label(state):
#     result = int(solver.lookup(state))
#     if result == 1:
#         label = -1.0
#     elif result == 2:
#         label = 1.0
#     else:
#         label = 0.0
#     return label


# def get_all_actions(state):
#     moves = cc.getMovesForward(state)
#     moves_list = pw.ll_to_list(moves)
    
#     return moves_list


# def get_optimal_actions(state):
#     moves = cc.getMovesForward(state)
#     moves_list = pw.ll_to_list(moves)
#     optimal_list = []

#     for move in moves_list:
#         cc.ApplyMove(state, move)
#         label = get_label(state)
#         if label == 1:  # or label == 0:
#             optimal_list.append(move)
#         cc.UndoMove(state, move)

#     return optimal_list
    

# def get_single_optimal_action(state):
#     moves = cc.getMoves(state)
#     moves_list = pw.ll_to_list(moves)
#     optimal_list = []

#     for move in moves_list:
#         cc.ApplyMove(state, move)
#         label = get_label(state)
#         if label == 1:  # or label == 0:
#             optimal_list.append(move)
#         cc.UndoMove(state, move)

#     return [random.choice(optimal_list)]


# def get_single_random_action(state):
#     moves = cc.getMovesForward(state)
#     moves_list = pw.ll_to_list(moves)

#     return [random.choice(moves_list)]


# def board_to_string(board):
#     state_str = ""
#     for x in board:
#         state_str += str(x)
#     return state_str


# def get_optimal_opp_action(state):
#     moves = cc.getMoves(state)
#     moves_list = pw.ll_to_list(moves)
#     labels = []
#     optimal_list = []

#     for move in moves_list:
#         cc.ApplyMove(state, move)
#         label = get_label(state)
#         labels.append(label)
#         cc.UndoMove(state, move)

#     best_val = min(labels)

#     for ind in range(len(labels)):
#         if labels[ind] == best_val:
#             optimal_list.append(moves_list[ind])

#     return [random.choice(optimal_list)]


# def get_smart_option(state):
#     moves = cc.getMovesForward(state)
#     moves_list = pw.ll_to_list(moves)
#     proportions = []

#     for move in moves_list:
#         cc.ApplyMove(state, move)
#         possible_moves = cc.getMovesForward(state)
#         p_moves_list = pw.ll_to_list(possible_moves)
#         p1_sub_optimals = 0
#         p1_outcomes = []

#         for p_move in p_moves_list:
#             cc.ApplyMove(state, p_move)
#             label = get_label(state)
#             p1_outcomes.append(label)
#             if label != 1:
#                 p1_sub_optimals += 1
#             cc.UndoMove(state, p_move)

#         proportions.append(p1_sub_optimals / len(p_moves_list))

#         # print(p1_outcomes)

#         cc.UndoMove(state, move)

#     # z = np.sum(proportions)
#     # if z != 0:
#     #     proportions = np.array(proportions) / np.sum(proportions)
#     # else:
#     #     proportions += np.ones_like(proportions) / np.size(proportions)

#     # print(proportions)

#     max_prop = 0
#     max_inds = []
#     for ind, prop in enumerate(proportions):
#         if prop > max_prop:
#             max_prop = prop
#             max_inds = [ind]
#         elif prop == max_prop:
#             max_inds.append(ind)

#     # print("choice: ", random.choice(max_inds))
#     return [moves_list[random.choice(max_inds)]]
#     # print(proportions)
#     # print(moves_list)
#     # print(proportions)
#     # return [np.random.choice(moves_list, p=proportions)]




# # moves = get_all_actions(game_state)
# # print(game_state.getBoard())
# # for move in moves:
# #     print(move.getFrom(), "->", move.getTo())




# def traverse(state=None, trajectory=[], num_trajectories=0, num_p1=0, p1=None):
#     if state is None:
#         state = cw.CCState()
#         cc.Reset(state)
#     if cc.Done(state):
#         # print(cc.Winner(state))
#         num_trajectories += 1
#         num_p1 = cc.Winner(state)
#         # output_file = open("/home/zaheen/projects/cc_games/4-4-3-mf/opt/-1/{}.json".format(num_trajectories), 'w')
#         # print(num_trajectories, cc.Winner(state))
#         output_data = trajectory + [str(cc.Winner(state))]
#         # json.dump(output_data, output_file)
#         # output_file.close()
#         # print(trajectory)
#         return num_trajectories, num_p1
#     else:
#         if not state.getToMove():
#             # print(state.getToMove())
            
#             # moves = get_optimal_actions(state)
#             moves = get_single_optimal_action(state)
#             # moves = get_single_random_action(state)
#             # moves, stats = p1.runMCTS(state, 0)
#             # moves = [moves]
#         else:
#             # moves = get_single_random_action(state)
#             # moves = get_smart_option(state)
#             moves = get_optimal_opp_action(state)

#         for move in moves:
#             cc.ApplyMove(state, move)
#             if board_to_string(state.getBoard()) not in trajectory:
#                 trajectory.append(board_to_string(state.getBoard()))
#                 num_trajectories, num_p1 = traverse(state, trajectory, num_trajectories, num_p1, p1)
#                 trajectory.pop(-1)
#             cc.UndoMove(state, move)
#             cc.delMove(move)
#         return num_trajectories, num_p1

# model = '/Users/bigyankarki/Desktop/bigyan/cc/results/async_jax/data/iteration_0/mod_0.npy'
# nnp = nnp.NNPlayer(model, 0)
# p1 = mcts.MCTS(cc, nnp)

# for i in range(1024):
#     print("here")
#     # cc.Reset(game_state)
#     # num_trajectories, num_p1 = traverse(game_state, [board_to_string(game_state.getBoard())], i, p1=p1)
#     # print(num_trajectories, num_p1)