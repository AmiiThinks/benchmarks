# Copyright (C) 2021-2022, Ken Tjhia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import torch as to

from domains import Domain, Problem, State
from enums import FourDir
from search import SearchNode, Trajectory, try_make_solution


class SlidingTilePuzzleState(State):
    def __init__(
        self,
        tiles: np.ndarray,
        blank_row: int,
        blank_col: int,
    ):
        self.tiles = tiles
        self.blank_row = blank_row
        self.blank_col = blank_col

    def __repr__(self) -> str:
        mlw = self.tiles.shape[0] ** 2
        return (
            f" {np.array2string(self.tiles, separator=' ' , max_line_width=mlw)[1:-1]}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return self.tiles.tobytes().__hash__()

    def __eq__(self, other) -> bool:
        return np.array_equal(self.tiles, other.tiles)


class SlidingTilePuzzle(Domain):
    def __init__(
        self, init_tiles: np.ndarray, goal_tiles: np.ndarray, forward: bool = True
    ):
        super().__init__(forward=forward)
        self.width = init_tiles.shape[0]
        self.num_tiles = self.width**2

        width_indices = np.arange(self.width)
        self._row_indices = np.repeat(width_indices, self.width)
        self._col_indices = np.tile(width_indices, self.width)

        blank_pos = np.where(init_tiles == 0)
        self.blank_row = blank_pos[0].item()
        self.blank_col = blank_pos[1].item()

        self.initial_state: SlidingTilePuzzleState = SlidingTilePuzzleState(
            init_tiles, self.blank_row, self.blank_col
        )

        self.initial_state_t: to.Tensor = self.state_tensor(self.initial_state)

        self.goal_tiles = goal_tiles
        blank_pos = np.where(goal_tiles == 0)
        goal_blank_row = blank_pos[0].item()
        goal_blank_col = blank_pos[1].item()

        self.goal_state = SlidingTilePuzzleState(
            goal_tiles, goal_blank_row, goal_blank_col
        )
        self.goal_state_t = self.state_tensor(self.goal_state)

    @property
    def try_make_solution_func(
        cls,
    ) -> Callable[
        [Domain, SearchNode, Domain, int], Optional[tuple[Trajectory, Trajectory]]
    ]:
        return try_make_solution

    @property
    def state_width(self) -> int:
        return self.width

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return self.num_tiles

    def state_tensor(
        self,
        state: SlidingTilePuzzleState,
    ) -> to.Tensor:
        arr = np.zeros((self.num_tiles, self.width, self.width), dtype=np.float32)
        indices = state.tiles.reshape(-1)
        arr[
            indices,
            self._row_indices,
            self._col_indices,
        ] = 1
        return to.from_numpy(arr)

    def reverse_action(self, action: FourDir) -> FourDir:
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def backward_domain(self) -> SlidingTilePuzzle:
        assert self.forward
        init_tiles = self.goal_tiles
        goal_tiles = self.initial_state.tiles
        domain = SlidingTilePuzzle(init_tiles, goal_tiles, forward=False)

        return domain

    def actions(
        self, parent_action: FourDir, state: SlidingTilePuzzleState
    ) -> list[FourDir]:
        actions = []

        if parent_action != FourDir.LEFT and state.blank_col != self.width - 1:
            actions.append(FourDir.RIGHT)

        if parent_action != FourDir.DOWN and state.blank_row != 0:
            actions.append(FourDir.UP)

        if parent_action != FourDir.RIGHT and state.blank_col != 0:
            actions.append(FourDir.LEFT)

        if parent_action != FourDir.UP and state.blank_row != self.width - 1:
            actions.append(FourDir.DOWN)

        return actions

    def actions_unpruned(self, state: SlidingTilePuzzleState):
        actions = []

        if state.blank_col != self.width - 1:
            actions.append(FourDir.RIGHT)

        if state.blank_row != 0:
            actions.append(FourDir.UP)

        if state.blank_col != 0:
            actions.append(FourDir.LEFT)

        if state.blank_row != self.width - 1:
            actions.append(FourDir.DOWN)

        return actions

    def result(
        self, state: SlidingTilePuzzleState, action: FourDir
    ) -> SlidingTilePuzzleState:
        tiles = np.array(state.tiles)
        blank_row = state.blank_row
        blank_col = state.blank_col

        if action == FourDir.UP:
            tiles[blank_row, blank_col], tiles[blank_row - 1, blank_col] = (
                tiles[blank_row - 1, blank_col],
                tiles[blank_row, blank_col],
            )
            blank_row -= 1

        elif action == FourDir.DOWN:
            tiles[blank_row, blank_col], tiles[blank_row + 1, blank_col] = (
                tiles[blank_row + 1, blank_col],
                tiles[blank_row, blank_col],
            )
            blank_row += 1

        elif action == FourDir.RIGHT:
            tiles[blank_row, blank_col], tiles[blank_row, blank_col + 1] = (
                tiles[blank_row, blank_col + 1],
                tiles[blank_row, blank_col],
            )
            blank_col += 1

        elif action == FourDir.LEFT:
            tiles[blank_row, blank_col], tiles[blank_row, blank_col - 1] = (
                tiles[blank_row, blank_col - 1],
                tiles[blank_row, blank_col],
            )
            blank_col -= 1

        new_state = SlidingTilePuzzleState(tiles, blank_row, blank_col)
        return new_state

    def is_goal(self, state: SlidingTilePuzzleState) -> bool:
        return np.array_equal(state.tiles, self.goal_tiles)


def parse_problemset(problemset: dict):
    width = problemset["width"]
    goal_tiles = np.arange(width**2).reshape(width, width)

    def parse_specs(problem_specs):
        problems = []
        for spec in problem_specs:
            init_tiles = np.array(spec["tiles"])
            problem = Problem(
                id=spec["id"],
                domain=SlidingTilePuzzle(init_tiles=init_tiles, goal_tiles=goal_tiles),
            )
            problems.append(problem)
        return problems

    model_args = {
        "num_actions": problemset["num_actions"],
        "in_channels": problemset["in_channels"],
        "state_t_width": problemset["state_t_width"],
        "double_backward": True,
    }

    if "is_curriculum" in problemset:
        bootstrap_problems = parse_specs(problemset["bootstrap_problems"])
        problemset["bootstrap_problems"] = bootstrap_problems
        problemset["curriculum_problems"] = parse_specs(
            problemset["curriculum_problems"]
        )
        problemset["permutation_problems"] = parse_specs(
            problemset["permutation_problems"]
        )
    else:
        problems = parse_specs(problemset["problems"])
        problemset["problems"] = problems

    return problemset, model_args
