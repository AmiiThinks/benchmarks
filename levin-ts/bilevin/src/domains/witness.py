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
from collections import deque
from copy import deepcopy
from typing import Callable, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch as to

from domains import Domain, Problem, State
from enums import Color, FourDir
from search import SearchNode, Trajectory


class WitnessState(State):
    """
    A Witness State.
    Note that start/head/goal row/col's refer to the grid, not cell locations
    Note that head_row and head_col are only consistent with grids/segs upon initialization if they're 0
    """

    def __init__(
        self,
        width: int,
        head_init_row,
        head_init_col,
        init_structs=True,
    ):
        self.head_row = head_init_row
        self.head_col = head_init_col
        self.width = width  # width of cells

        if init_structs:
            self.grid = np.zeros((self.width + 1, self.width + 1))
            self.grid[self.head_row, self.head_col] = 1

            self.v_segs = np.zeros((self.width, self.width + 1))
            self.h_segs = np.zeros((self.width + 1, self.width))

    def __hash__(self) -> int:
        """
        Note that these hash and eq implementations imply the states are generated
        from the same problem
        """
        return (
            self.h_segs.tobytes(),
            self.v_segs.tobytes(),
        ).__hash__()

    def __eq__(self, other) -> bool:
        return (
            np.array_equal(self.grid, other.grid)
            and np.array_equal(self.h_segs, other.h_segs)
            and np.array_equal(self.v_segs, other.v_segs)
        )


def get_merged_trajectory(
    dir1_domain,
    dir1_common: SearchNode,
    dir2_common: SearchNode,
    node_type: Type[SearchNode],
    num_expanded: int,
    goal_state_t: Optional[to.Tensor] = None,
    forward: bool = True,
) -> Trajectory:
    """
    Returns a new trajectory going from dir1_start to dir2_start, passing through
    merge(dir1_common, dir2_common).
    """
    dir1_node = dir1_common
    dir2_parent_node = dir2_common.parent
    dir2_parent_action = dir2_common.parent_action
    while dir2_parent_node:
        action = dir1_domain.reverse_action(dir2_parent_action)
        new_state = dir1_domain.result(dir1_node.state, action)
        new_dir1_node = node_type(
            state=new_state,
            parent=dir1_node,
            parent_action=action,
            g_cost=dir1_node.g_cost + 1,
        )
        dir1_node = new_dir1_node
        dir2_parent_action = dir2_parent_node.parent_action
        dir2_parent_node = dir2_parent_node.parent

    return Trajectory(dir1_domain, dir1_node, num_expanded, goal_state_t, forward)


def try_make_solution(
    this_domain: Witness, node: SearchNode, other_domain: Witness, num_expanded: int
) -> Optional[tuple[Trajectory, Trajectory]]:
    """
    Tries to create a solution from the current node and the nodes visited by the other search
    direction. Check if the heads coincide, and if so create a single node with its trajectory
    in the direction specified by self. Check if the merged state is a goal state and create a
    forward and backward Trajectory if so.

    """
    state: WitnessState = node.state
    head_dot = (state.head_row, state.head_col)
    if head_dot not in other_domain.visited:
        return None
    for other_node in other_domain.visited[head_dot]:
        other_state = other_node.state

        merged_state = WitnessState(
            this_domain.width,
            other_domain.start_row,
            other_domain.start_col,
            init_structs=False,
        )

        merged_state.grid = state.grid + other_state.grid
        merged_state.grid[head_dot] = 1
        # todo is this sufficient and neccesry to prevent overlaps bessides at head?
        if np.any(merged_state.grid > 1.5):
            return None

        merged_state.v_segs = state.v_segs + other_state.v_segs
        merged_state.h_segs = state.h_segs + other_state.h_segs

        if this_domain.is_goal(merged_state):
            if this_domain.forward:
                f_common_node = node
                b_common_node = other_node
                f_domain = this_domain
                b_domain = other_domain
            else:
                f_common_node = other_node
                b_common_node = node
                f_domain = other_domain
                b_domain = this_domain

            f_traj = get_merged_trajectory(
                f_domain, f_common_node, b_common_node, type(node), num_expanded
            )
            b_traj = get_merged_trajectory(
                b_domain,
                b_common_node,
                f_common_node,
                type(node),
                num_expanded,
                forward=False,
            )
            return (f_traj, b_traj)

    return None


class Witness(Domain):
    """
    An executor for a Witness problem.
    """

    def __init__(
        self,
        width: int,
        max_num_colors: int,
        init: list[int],
        goal: list[int],
        colored_cells: list[str] = [],
        forward: bool = True,
    ):
        """
        Initializes  a witness executor specific to a problem, and creates the initial state.
        Note we hardcode max number of colors to 7 and max width to 10 cells (square problems only)

        Parameters
        ----------
        puzzle :
            A 3 element list of strings. The elements correspond to the Size, Init, Goal, and
            Colors, as
            specified in the witness puzzle dataset, or the generator script.
        """
        super().__init__(forward=forward)
        self.max_num_colors = max_num_colors
        self.width = width

        self.start_row = init[0]
        self.start_col = init[1]

        self.goal_row = goal[0]
        self.goal_col = goal[1]

        self.cells = np.zeros((self.width, self.width), dtype=np.int32)

        self.num_colors = 0
        for cell in colored_cells:
            values = [int(x) for x in cell.split()]
            self.cells[values[0], values[1]] = values[2]

        self._colored_idxs = [
            (i, j)
            for i in range(self.width)
            for j in range(self.width)
            if self.cells[i, j] != 0
        ]

        self.initial_state: WitnessState = WitnessState(
            self.width,
            self.start_row,
            self.start_col,
        )

    def update(self, node: SearchNode):
        state: WitnessState = node.state
        head = (state.head_row, state.head_col)
        if head in self.visited:
            self.visited[head].append(node)
        else:
            self.visited[head] = [node]

    @property
    def try_make_solution_func(
        cls,
    ) -> Callable[
        [Witness, SearchNode, Witness, int], Optional[tuple[Trajectory, Trajectory]]
    ]:
        return try_make_solution

    @property
    def double_backward(self):
        return False

    @property
    def num_actions(cls) -> int:
        return 4

    @property
    def in_channels(self) -> int:
        """
        The max num of colors any problem in a particular problem set may have.
        """
        return self.max_num_colors + 5

    @property
    def state_width(self) -> int:
        return self.width + 1

    def state_tensor(self, state: WitnessState) -> to.Tensor:
        """
        Generates an image representation for the puzzle. one channel for each color; one channel with 1's
        where is "open" in the grid (this allows learning systems to work with a fixed image size defined
        by max_lines and max_columns); one channel for the current path (cells occupied by the snake);
        one channel for the tip of the snake; one channel for the exit of the puzzle; one channel for the
        entrance of the snake. In total there are 9 different channels.

        Each channel is a matrix with zeros and ones. The image returned is a 3-dimensional numpy array.
        """

        # defining the 3-dimnesional array that will be filled with the puzzle's information
        arr = np.zeros(
            (self.in_channels, self.state_width, self.state_width), dtype=np.float32
        )

        for i in range(self.width):
            for j in range(self.width):
                color = self.cells[i, j]
                if color != 0:
                    arr[
                        color - 1, i, j
                    ] = 1  # -1 because we don't encode the neutral color

        channel_number = self.max_num_colors
        # channels for the current path
        # vsegs
        for i in range(self.width):
            for j in range(self.width + 1):
                if state.v_segs[i, j] == 1:
                    arr[channel_number, i, j] = 1

        channel_number += 1
        # hsegs
        for i in range(self.width + 1):
            for j in range(self.width):
                if state.h_segs[i, j] == 1:
                    arr[channel_number, i, j] = 1

        # channel with the tip of the snake
        channel_number += 1
        arr[channel_number, state.head_row, state.head_col] = 1

        # channel for the exit of the puzzle
        channel_number += 1
        arr[channel_number, self.goal_row, self.goal_col] = 1

        # channel for the entrance of the puzzle
        channel_number += 1
        arr[channel_number, self.start_row, self.start_col] = 1

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

    def actions(self, parent_action: FourDir, state: WitnessState) -> list[FourDir]:
        """
        Successor function used by planners trying to solve the puzzle. The method returns
        a list with legal actions for the state. The valid actions for the domain are {U, D, L, R}.

        The tip of the snake can move to an adjacent intersection in the grid as long as
        that intersection isn't already occupied by the snake and the intersection is valid
        (i.e., it isn't negative or larger than the grid size)

        op is the action taken at the parent; used here to perform parent pruning

        Mapping of actions:
        0 - Up
        1 - Down
        2 - Right
        3 - Left
        """
        actions = []
        # moving up
        if (
            parent_action != FourDir.DOWN
            and state.head_row + 1 < state.grid.shape[0]
            and state.grid[state.head_row + 1, state.head_col] == 0
        ):
            actions.append(FourDir.UP)
        # moving down
        if (
            parent_action != FourDir.UP
            and state.head_row >= 1
            and state.grid[state.head_row - 1, state.head_col] == 0
        ):
            actions.append(FourDir.DOWN)
        # moving right
        if (
            parent_action != FourDir.LEFT
            and state.head_col + 1 < state.grid.shape[1]
            and state.grid[state.head_row, state.head_col + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        # moving left
        if (
            parent_action != FourDir.RIGHT
            and state.head_col >= 1
            and state.grid[state.head_row, state.head_col - 1] == 0
        ):
            actions.append(FourDir.LEFT)

        return actions

    def actions_unpruned(self, state: WitnessState) -> list[FourDir]:
        actions = []
        # moving up
        if (
            state.head_row + 1 < state.grid.shape[0]
            and state.grid[state.head_row + 1, state.head_col] == 0
        ):
            actions.append(FourDir.UP)
        # moving down
        if state.head_row >= 1 and state.grid[state.head_row - 1, state.head_col] == 0:
            actions.append(FourDir.DOWN)
        # moving right
        if (
            state.head_col + 1 < state.grid.shape[1]
            and state.grid[state.head_row, state.head_col + 1] == 0
        ):
            actions.append(FourDir.RIGHT)
        # moving left
        if state.head_col >= 1 and state.grid[state.head_row, state.head_col - 1] == 0:
            actions.append(FourDir.LEFT)

        return actions

    def result(self, state: WitnessState, action: FourDir) -> WitnessState:
        """
        Applies a given action to the state. It modifies the segments visited by the snake (v_seg and h_seg),
        the intersections (grid), and the tip of the snake.
        """
        # faster than deepcopy or np.ndarray.copy()
        # todo make structs State constructor params
        new_state = WitnessState(
            self.width, state.head_row, state.head_col, init_structs=False
        )
        new_state.grid = np.array(state.grid)
        new_state.v_segs = np.array(state.v_segs)
        new_state.h_segs = np.array(state.h_segs)

        # moving up
        if action == FourDir.UP:
            new_state.v_segs[new_state.head_row, new_state.head_col] = 1
            new_state.grid[new_state.head_row + 1, new_state.head_col] = 1
            new_state.head_row += 1
        # moving down
        elif action == FourDir.DOWN:
            new_state.v_segs[new_state.head_row - 1, new_state.head_col] = 1
            new_state.grid[new_state.head_row - 1, new_state.head_col] = 1
            new_state.head_row -= 1
        # moving right
        elif action == FourDir.RIGHT:
            new_state.h_segs[new_state.head_row, new_state.head_col] = 1
            new_state.grid[new_state.head_row, new_state.head_col + 1] = 1
            new_state.head_col += 1
        # moving left
        elif action == FourDir.LEFT:
            new_state.h_segs[new_state.head_row, new_state.head_col - 1] = 1
            new_state.grid[new_state.head_row, new_state.head_col - 1] = 1
            new_state.head_col -= 1

        return new_state

    def is_head_at_goal(self, state: WitnessState) -> bool:
        return self.goal_row == state.head_row and self.goal_col == state.head_col

    def is_goal(self, state: WitnessState) -> bool:
        """
        Verifies whether the state's path represents a valid solution. This is performed by verifying the following
        (1) the tip of the snake is at the goal position
        (2) a bullet of color c1 cannot reach a bullet of color c2 through a BFS search.

        The BFS uses the cells (line and column) as states and verifies whether cells with a bullet of a given color
        can only reach (and be reached) by cells with bullets of the same color (or of the neutral color, denoted as zero in this implementation)
        """
        if not self.is_head_at_goal(state):
            return False

        reached = set()

        for root in self._colored_idxs:
            # If root of new BFS search was already visited, then go to the next state
            if root in reached:
                continue
            current_color = self.cells[root]

            frontier = deque()
            frontier.append(root)
            reached.add(root)
            while len(frontier) != 0:
                cell = frontier.popleft()

                def reachable_neighbors(self, cell) -> list[tuple[int, int]]:
                    """
                     Breadth-first search (BFS) performed to validate a solution.
                    An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg)
                    separating cells c and c'.

                    """
                    neighbors = []
                    row, col = cell
                    # move up
                    if row + 1 < self.width and state.h_segs[row + 1, col] == 0:
                        neighbors.append((row + 1, col))
                    # move down
                    if row > 0 and state.h_segs[row, col] == 0:
                        neighbors.append((row - 1, col))
                    # move right
                    if col + 1 < self.width and state.v_segs[row, col + 1] == 0:
                        neighbors.append((row, col + 1))
                    # move left
                    if col > 0 and state.v_segs[row, col] == 0:
                        neighbors.append((row, col - 1))
                    return neighbors

                neighbors = reachable_neighbors(self, cell)
                for neighbor in neighbors:
                    if neighbor in reached:
                        continue
                    if (
                        self.cells[neighbor] != 0
                        and self.cells[neighbor] != current_color
                    ):
                        return False
                    frontier.append(neighbor)
                    reached.add(neighbor)
        return True

    def backward_domain(self) -> Witness:
        """
        Should only be called on a fresh domain (no calls to update). Reverses a witness problem by
        reversing the head and goal (and updating grid to be consistent with this change).
        """
        assert self.forward
        assert len(self.visited) == 0

        b_domain = deepcopy(self)

        state = b_domain.initial_state
        # b_domain.goal_state_t = b_domain.state_tensor(state)
        b_domain.forward = False

        state.grid[self.start_row, self.start_col] = 0
        state.grid[self.goal_row, self.goal_col] = 1

        state.head_row = self.goal_row
        state.head_col = self.goal_col

        b_domain.start_row = self.goal_row
        b_domain.start_col = self.goal_col

        b_domain.goal_row = self.start_row
        b_domain.goal_col = self.start_col

        return b_domain

    def __repr__(self) -> str:
        return self.initial_state.__repr__()

    def plot(self, state: Optional[WitnessState] = None, filename=None):
        """
        This method plots the state. Several features in this method are hard-coded and might
        need adjustment as one changes the size of the puzzle. For example, the size of the figure is set to be fixed
        to [5, 5] (see below).
        """
        if not state:
            state = self.initial_state

        ax: plt.Axes
        _, ax = plt.subplots(figsize=(5, 5))
        # fig.patch.set_facecolor((1, 1, 1))

        # draw vertical lines of the grid
        for y in range(state.grid.shape[1]):
            ax.plot([y, y], [0, self.width], str(Color.BLACK))
        # draw horizontal lines of the grid
        for x in range(state.grid.shape[0]):
            ax.plot([0, self.width], [x, x], str(Color.BLACK))

        # scale the axis area to fill the whole figure
        ax.set_position([0, 0, 1, 1])

        ax.set_axis_off()

        ax.set_xlim(-1, np.max(state.grid.shape))
        ax.set_ylim(-1, np.max(state.grid.shape))

        # Draw the vertical segments of the path
        for i in range(state.v_segs.shape[0]):
            for j in range(state.v_segs.shape[1]):
                if state.v_segs[i, j] == 1:
                    ax.plot([j, j], [i, i + 1], str(Color.RED), linewidth=5)

        # Draw the horizontal segments of the path
        for i in range(state.h_segs.shape[0]):
            for j in range(state.h_segs.shape[1]):
                if state.h_segs[i, j] == 1:
                    ax.plot([j, j + 1], [i, i], str(Color.RED), linewidth=5)

        # Draw the separable bullets according to the values in self.cells and Color enum type
        offset = 0.5
        color_strings = Color.str_values()[1:]
        for i in range(self.width):
            for j in range(self.width):
                if self.cells[i, j] != 0:
                    ax.plot(
                        j + offset,
                        i + offset,
                        "o",
                        markersize=15,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=color_strings[int(self.cells[i, j] - 1)],
                        markeredgewidth=2,
                    )

        # Draw the intersection of lines: red for an intersection that belongs to a path and black otherwise
        for i in range(state.grid.shape[0]):
            for j in range(state.grid.shape[1]):
                if state.grid[i, j] != 0:
                    ax.plot(
                        j,
                        i,
                        "o",
                        markersize=10,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=str(Color.RED),
                        markeredgewidth=0,
                    )
                else:
                    ax.plot(
                        j,
                        i,
                        "o",
                        markersize=10,
                        markeredgecolor=(0, 0, 0),
                        markerfacecolor=str(Color.BLACK),
                        markeredgewidth=0,
                    )

        # Draw the entrance of the puzzle in red as it is always on the state's path
        ax.plot(
            self.start_col - 0.15,
            self.start_row,
            ">",
            markersize=10,
            markeredgecolor=(0, 0, 0),
            markerfacecolor=str(Color.RED),
            markeredgewidth=0,
        )

        column_exit_offset = 0
        row_exit_offset = 0

        if self.goal_col == self.width:
            column_exit_offset = 0.15
            exit_symbol = ">"
        elif self.goal_col == 0:
            column_exit_offset = -0.15
            exit_symbol = "<"
        elif self.goal_row == self.width:
            row_exit_offset = 0.15
            exit_symbol = "^"
        else:
            row_exit_offset = -0.15
            exit_symbol = "v"
        # Draw the exit of the puzzle: red if it is on a path, black otherwise
        if state.grid[self.goal_row, self.goal_col] == 0:
            ax.plot(
                self.goal_col + column_exit_offset,
                self.goal_row + row_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor=str(Color.BLACK),
                markeredgewidth=0,
            )
        else:
            ax.plot(
                self.goal_col + column_exit_offset,
                self.goal_row + row_exit_offset,
                exit_symbol,
                markersize=10,
                markeredgecolor=(0, 0, 0),
                markerfacecolor=str(Color.RED),
                markeredgewidth=0,
            )

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


def parse_problemset(problemset: dict):
    width = problemset["width"]
    max_num_colors = problemset["max_num_colors"]

    def parse_specs(problem_specs):
        problems = []
        for spec in problem_specs:
            problem = Problem(
                id=spec["id"],
                domain=Witness(
                    width=width,
                    max_num_colors=max_num_colors,
                    init=spec["init"],
                    goal=spec["goal"],
                    colored_cells=spec["colored_cells"],
                ),
            )
            problems.append(problem)
        return problems

    model_args = {
        "num_actions": problemset["num_actions"],
        "in_channels": problemset["in_channels"],
        "state_t_width": problemset["state_t_width"],
        "double_backward": False,
    }

    if "is_curriculum" in problemset:
        problems = parse_specs(problemset["curriculum_problems"])
        problemset["curriculum_problems"] = problems
    else:
        problems = parse_specs(problemset["problems"])
        problemset["problems"] = problems

    return problemset, model_args
