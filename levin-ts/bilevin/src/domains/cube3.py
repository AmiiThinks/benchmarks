"""
Modified from https://github.com/forestagostinelli/DeepCubeA/blob/master/environments/cube3.py,
the code for the paper 'Solving the Rubik’s cube with deep reinforcement learning and search'
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import torch as to
from torch import nn

from domains import Domain, State


class Cube3State(State):
    __slots__ = ["colors", "hash"]

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors

    def __str__(self) -> str:
        return str(self.colors)

    def __hash__(self) -> int:
        return self.colors.tobytes().__hash__()

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube3(Domain):
    moves: List[str] = [
        "%s%i" % (f, n) for f in ["U", "D", "L", "R", "B", "F"] for n in [-1, 1]
    ]
    moves_rev: List[str] = [
        "%s%i" % (f, n) for f in ["U", "D", "L", "R", "B", "F"] for n in [1, -1]
    ]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # solved state
        self.goal_colors: np.ndarray = np.arange(
            0, (self.cube_len**2) * 6, 1, dtype=self.dtype
        )

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(
            self.cube_len, self.moves
        )

    @property
    def state_width(self) -> int:
        pass

    @property
    def num_actions(cls) -> int:
        return 12

    @property
    def in_channels(self) -> int:
        pass

    def result(self, state: Cube3State, action: int) -> Cube3State:
        next_state_np = self._move_np(state.colors, action)
        next_state: Cube3State = Cube3State(next_state_np)

        return next_state

    # todo will be used to compute reverse result
    def get_backward_domain(self) -> Cube3:
        pass

    def prev_state(self, states: List[Cube3State], action: int) -> List[Cube3State]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.result(states, move_rev_idx)[0]

    def is_goal(self, state: Cube3State) -> np.ndarray:
        is_equal = np.equal(state.colors, self.goal_colors)

        return np.all(is_equal, axis=1)

    def state_tensor(self, state: Cube3State) -> to.Tensor:
        state_t: to.Tensor = to.tensor(state.colors) / (self.cube_len**2)
        return state_t

    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]

        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[
            :, self.rotate_idxs_old[action_str]
        ]

        return states_next_np

    def _get_adj(self) -> None:
        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.adj_faces: Dict[int, np.ndarray] = {
            0: np.array([2, 5, 3, 4]),
            1: np.array([2, 4, 3, 5]),
            2: np.array([0, 4, 1, 5]),
            3: np.array([0, 5, 1, 4]),
            4: np.array([0, 3, 1, 2]),
            5: np.array([0, 2, 1, 3]),
        }

    def _compute_rotation_idxs(
        self, cube_len: int, moves: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {
                0: {
                    2: [range(0, cube_len), cube_len - 1],
                    3: [range(0, cube_len), cube_len - 1],
                    4: [range(0, cube_len), cube_len - 1],
                    5: [range(0, cube_len), cube_len - 1],
                },
                1: {
                    2: [range(0, cube_len), 0],
                    3: [range(0, cube_len), 0],
                    4: [range(0, cube_len), 0],
                    5: [range(0, cube_len), 0],
                },
                2: {
                    0: [0, range(0, cube_len)],
                    1: [0, range(0, cube_len)],
                    4: [cube_len - 1, range(cube_len - 1, -1, -1)],
                    5: [0, range(0, cube_len)],
                },
                3: {
                    0: [cube_len - 1, range(0, cube_len)],
                    1: [cube_len - 1, range(0, cube_len)],
                    4: [0, range(cube_len - 1, -1, -1)],
                    5: [cube_len - 1, range(0, cube_len)],
                },
                4: {
                    0: [range(0, cube_len), cube_len - 1],
                    1: [range(cube_len - 1, -1, -1), 0],
                    2: [0, range(0, cube_len)],
                    3: [cube_len - 1, range(cube_len - 1, -1, -1)],
                },
                5: {
                    0: [range(0, cube_len), 0],
                    1: [range(cube_len - 1, -1, -1), cube_len - 1],
                    2: [cube_len - 1, range(0, cube_len)],
                    3: [0, range(cube_len - 1, -1, -1)],
                },
            }
            face_dict = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[
                    (np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to)))
                    % len(faces_to)
                ]

            cubes_idxs = [
                [0, range(0, cube_len)],
                [range(0, cube_len), cube_len - 1],
                [cube_len - 1, range(cube_len - 1, -1, -1)],
                [range(cube_len - 1, -1, -1), 0],
            ]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[
                    (np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to)))
                    % len(cubes_to)
                ]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [
                    [idx1, idx2]
                    for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten()
                    for idx2 in np.array([cubes_idxs[cubes_to[i]][1]]).flatten()
                ]
                idxs_old = [
                    [idx1, idx2]
                    for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten()
                    for idx2 in np.array([cubes_idxs[cubes_from[i]][1]]).flatten()
                ]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index(
                        (face, idxNew[0], idxNew[1]), colors_new.shape
                    )
                    flat_idx_old = np.ravel_multi_index(
                        (face, idxOld[0], idxOld[1]), colors.shape
                    )
                    rotate_idxs_new[move] = np.concatenate(
                        (rotate_idxs_new[move], [flat_idx_new])
                    )
                    rotate_idxs_old[move] = np.concatenate(
                        (rotate_idxs_old[move], [flat_idx_old])
                    )

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [
                    [idx1, idx2]
                    for idx1 in np.array([face_idxs[face_to][0]]).flatten()
                    for idx2 in np.array([face_idxs[face_to][1]]).flatten()
                ]
                idxs_old = [
                    [idx1, idx2]
                    for idx1 in np.array([face_idxs[face_from][0]]).flatten()
                    for idx2 in np.array([face_idxs[face_from][1]]).flatten()
                ]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index(
                        (face_to, idxNew[0], idxNew[1]), colors_new.shape
                    )
                    flat_idx_old = np.ravel_multi_index(
                        (face_from, idxOld[0], idxOld[1]), colors.shape
                    )
                    rotate_idxs_new[move] = np.concatenate(
                        (rotate_idxs_new[move], [flat_idx_new])
                    )
                    rotate_idxs_old[move] = np.concatenate(
                        (rotate_idxs_old[move], [flat_idx_old])
                    )

        return rotate_idxs_new, rotate_idxs_old
