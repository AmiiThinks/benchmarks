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

from domains.domain import Domain, Problem, State
from enums import FourDir
from search import SearchNode, Trajectory, try_make_solution


class SokobanState(State):
    def __init__(self, man_row: int, man_col: int, boxes: np.ndarray) -> None:
        self.man_row = man_row
        self.man_col = man_col
        self.boxes = boxes

    def __eq__(self, other) -> bool:
        return (
            self.man_row == other.man_row
            and self.man_col == other.man_col
            and np.array_equal(self.boxes, other.boxes)
        )

    def __hash__(self) -> int:
        return (self.man_row, self.man_col, self.boxes.tobytes()).__hash__()


class Sokoban(Domain):
    goal_str = "."
    man_str = "@"
    wall_str = "#"
    box_str = "$"

    wall_channel = 0
    goal_channel = 1
    box_channel = 2
    man_channel = 3

    def __init__(
        self,
        map: np.ndarray,
        man_row: int,
        man_col: int,
        boxes: np.ndarray,
        forward: bool = True,
    ) -> None:
        super().__init__(forward=forward)

        self.map = map
        self.rows = map.shape[1]
        self.cols = map.shape[2]

        self.original_boxes = boxes
        self.initial_state = SokobanState(man_row, man_col, boxes)
        self.initial_state_t = self.state_tensor(self.initial_state)

        self.goal_state: SokobanState
        self.goal_state_t: to.Tensor

    @property
    def try_make_solution_func(
        cls,
    ) -> Callable[
        [Domain, SearchNode, Domain, int], Optional[tuple[Trajectory, Trajectory]]
    ]:
        return try_make_solution

    @property
    def state_width(cls) -> int:
        return 10

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        return 4

    def actions_unpruned(self, state: SokobanState) -> list[FourDir]:
        """
        Can move in each direction where either there is no wall or box, or there is a box that can
        be pushed (i.e. no wall or box behind it)
        """
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if (
            self.map[Sokoban.wall_channel, man_row, man_col + 1] == 0
            and state.boxes[man_row, man_col + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        elif (
            state.boxes[man_row, man_col + 1] == 1
            and self.map[Sokoban.wall_channel, man_row, man_col + 2] == 0
            and state.boxes[man_row, man_col + 2] == 0
        ):
            actions.append(FourDir.RIGHT)

        if (
            self.map[Sokoban.wall_channel, man_row, man_col - 1] == 0
            and state.boxes[man_row, man_col - 1] == 0
        ):
            actions.append(FourDir.LEFT)
        elif (
            state.boxes[man_row, man_col - 1] == 1
            and self.map[Sokoban.wall_channel, man_row, man_col - 2] == 0
            and state.boxes[man_row, man_col - 2] == 0
        ):
            actions.append(FourDir.LEFT)

        if (
            self.map[Sokoban.wall_channel, man_row + 1, man_col] == 0
            and state.boxes[man_row + 1, man_col] == 0
        ):
            actions.append(FourDir.DOWN)
        elif (
            state.boxes[man_row + 1, man_col] == 1
            and self.map[Sokoban.wall_channel, man_row + 2, man_col] == 0
            and state.boxes[man_row + 2, man_col] == 0
        ):
            actions.append(FourDir.DOWN)

        if (
            self.map[Sokoban.wall_channel, man_row - 1, man_col] == 0
            and state.boxes[man_row - 1, man_col] == 0
        ):
            actions.append(FourDir.UP)
        elif (
            state.boxes[man_row - 1, man_col] == 1
            and self.map[Sokoban.wall_channel, man_row - 2, man_col] == 0
            and state.boxes[man_row - 2, man_col] == 0
        ):
            actions.append(FourDir.UP)

        return actions

    def actions(self, parent_action: FourDir, state: SokobanState) -> list[FourDir]:
        return self.actions_unpruned(state)

    def result(self, state: SokobanState, action: FourDir) -> SokobanState:
        boxes = np.array(state.boxes)
        man_row = state.man_row
        man_col = state.man_col

        if action == FourDir.UP:
            if boxes[man_row - 1, man_col] == 1:
                boxes[man_row - 1, man_col] = 0
                boxes[man_row - 2, man_col] = 1
            man_row -= 1

        if action == FourDir.DOWN:
            if boxes[man_row + 1, man_col] == 1:
                boxes[man_row + 1, man_col] = 0
                boxes[man_row + 2, man_col] = 1
            man_row += 1

        if action == FourDir.RIGHT:
            if boxes[man_row, man_col + 1] == 1:
                boxes[man_row, man_col + 1] = 0
                boxes[man_row, man_col + 2] = 1
            man_col += 1

        if action == FourDir.LEFT:
            if boxes[man_row, man_col - 1] == 1:
                boxes[man_row, man_col - 1] = 0
                boxes[man_row, man_col - 2] = 1
            man_col -= 1

        return SokobanState(man_row, man_col, boxes)

    def reverse_action(self, action: FourDir) -> FourDir:
        if action == FourDir.UP:
            return FourDir.DOWN
        elif action == FourDir.DOWN:
            return FourDir.UP
        elif action == FourDir.LEFT:
            return FourDir.RIGHT
        elif action == FourDir.RIGHT:
            return FourDir.LEFT

    def _backward_actions_unpruned(self, state: SokobanState) -> list[FourDir]:
        assert not self.forward
        actions = []
        man_row = state.man_row
        man_col = state.man_col

        if (
            self.map[Sokoban.wall_channel, man_row, man_col + 1] == 0
            and state.boxes[man_row, man_col + 1] == 0
        ):
            actions.append(FourDir.RIGHT)

        if (
            self.map[Sokoban.wall_channel, man_row, man_col - 1] == 0
            and state.boxes[man_row, man_col - 1] == 0
        ):
            actions.append(FourDir.LEFT)

        if (
            self.map[Sokoban.wall_channel, man_row + 1, man_col] == 0
            and state.boxes[man_row + 1, man_col] == 0
        ):
            actions.append(FourDir.DOWN)

        if (
            self.map[Sokoban.wall_channel, man_row - 1, man_col] == 0
            and state.boxes[man_row - 1, man_col] == 0
        ):
            actions.append(FourDir.UP)

        return actions

    def _backward_actions(
        self, parent_action: FourDir, state: SokobanState
    ) -> list[FourDir]:
        return self._backward_actions_unpruned(state)

    def _backward_result(self, state: SokobanState, action: FourDir) -> SokobanState:
        boxes = np.array(state.boxes)
        man_row = state.man_row
        man_col = state.man_col

        if action == FourDir.UP:
            if boxes[man_row + 1, man_col] == 1:
                boxes[man_row + 1, man_col] = 0
                boxes[man_row - 1, man_col] = 1
            man_row -= 1

        if action == FourDir.DOWN:
            if boxes[man_row - 1, man_col] == 1:
                boxes[man_row - 1, man_col] = 0
                boxes[man_row + 1, man_col] = 1
            man_row += 1

        if action == FourDir.RIGHT:
            if boxes[man_row, man_col - 1] == 1:
                boxes[man_row, man_col - 1] = 0
                boxes[man_row, man_col + 1] = 1
            man_col += 1

        if action == FourDir.LEFT:
            if boxes[man_row, man_col + 1] == 1:
                boxes[man_row, man_col + 1] = 0
                boxes[man_row, man_col - 1] = 1
            man_col -= 1

        return SokobanState(man_row, man_col, boxes)

    def backward_domain(self) -> Sokoban:
        """
        create new states by placing the boxes on the goals, then the man at each side of each box, where not blocked
        """
        assert self.forward
        boxes = self.map[Sokoban.goal_channel]
        walls = self.map[Sokoban.wall_channel]
        new_domain = Sokoban(
            self.map, 0, 0, boxes, forward=False
        )  # temp man_row, man_col
        states = []
        box_coords = np.argwhere(boxes)
        for row, col in box_coords:
            row = int(row)
            col = int(col)
            if walls[row - 1, col] == 0 and boxes[row - 1, col] == 0:
                states.append(SokobanState(row - 1, col, boxes))

            if walls[row + 1, col] == 0 and boxes[row + 1, col] == 0:
                states.append(SokobanState(row + 1, col, boxes))

            if walls[row, col + 1] == 0 and boxes[row, col + 1] == 0:
                states.append(SokobanState(row, col + 1, boxes))

            if walls[row, col - 1] == 0 and boxes[row, col - 1] == 0:
                states.append(SokobanState(row, col - 1, boxes))

        del new_domain.initial_state
        new_domain.initial_state = states
        new_domain.initial_state_t = None
        new_domain.goal_state = self.initial_state
        new_domain.goal_state_t = self.initial_state_t
        new_domain.actions = new_domain._backward_actions
        new_domain.actions_unpruned = new_domain._backward_actions_unpruned
        new_domain.result = new_domain._backward_result
        new_domain.is_goal = new_domain._backward_is_goal

        return new_domain

    def _backward_is_goal(self, state: SokobanState) -> bool:
        assert not self.forward
        return state == self.goal_state

    def is_goal(self, state: SokobanState) -> bool:
        return np.array_equal(state.boxes, self.map[Sokoban.goal_channel])

    def state_tensor(self, state: SokobanState) -> to.Tensor:
        channel_man = np.zeros((self.rows, self.cols), dtype=np.float32)
        channel_man[state.man_row, state.man_col] = 1

        arr = np.concatenate(
            (self.map, state.boxes[None, ...], channel_man[None, ...]),
            axis=0,
            dtype=np.float32,
        )
        return to.from_numpy(arr)

    def print(self, state: SokobanState):
        for i in range(self.rows):
            for j in range(self.cols):
                if (
                    self.map[
                        Sokoban.goal_channel,
                        i,
                        j,
                    ]
                    == 1
                    and state.boxes[i, j] == 1
                ):
                    print("*", end="")
                elif i == state.man_row and j == state.man_col:
                    print(Sokoban.man_str, end="")
                elif (
                    self.map[
                        Sokoban.goal_channel,
                        i,
                        j,
                    ]
                    == 1
                ):
                    print(Sokoban.goal_str, end="")
                elif (
                    self.map[
                        Sokoban.wall_channel,
                        i,
                        j,
                    ]
                    == 1
                ):
                    print(Sokoban.wall_str, end="")
                elif state.boxes[i, j] == 1:
                    print(Sokoban.box_str, end="")
                else:
                    print(" ", end="")
            print()


def parse_problemset(problemset: dict):
    def parse_specs(problem_specs):
        problems = []
        for spec in problem_specs:
            problem = Problem(
                id=spec["id"],
                domain=Sokoban(
                    map=np.array(spec["map"], dtype=np.float32),
                    man_row=spec["man_row"],
                    man_col=spec["man_col"],
                    boxes=np.array(spec["boxes"], dtype=np.float32),
                ),
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
        problems = parse_specs(problemset["curriculum_problems"])
        problemset["curriculum_problems"] = problems
    else:
        problems = parse_specs(problemset["problems"])
        problemset["problems"] = problems

    return problemset, model_args
