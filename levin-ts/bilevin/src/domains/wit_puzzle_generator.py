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

import argparse
from collections import deque
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

from domains.witness import Witness


def main():
    """
    Generate a dataset of Witness problems. A generated problem instance is only kept if
    there least args.width // 2 colored regions of size at least 2, each with a unique color.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="directory path to save problem instances",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=4,
        help="width of puzzles to be generated",
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=50000,
        help="number of training puzzles to be generated",
    )
    parser.add_argument(
        "--n-valid",
        type=int,
        default=1000,
        help="number of validation puzzles to be generated",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="number of testing puzzles to be generated",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    parser.add_argument(
        "-c",
        "--max-num-colors",
        type=int,
        default=4,
        help="number of colors to use",
    )

    parser.add_argument(
        "-p",
        "--bullet-prob",
        type=float,
        default=0.6,
        help="probability of placing a bullet in each empty cell",
    )

    args = parser.parse_args()

    if args.max_num_colors < 2:
        raise ValueError("Number of colors must be at least 2")

    args.output_path.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    goals = (
        [(0, i) for i in range(args.width + 1)]
        + [(i, 0) for i in range(args.width + 1)]
        + [(args.width, i) for i in range(args.width + 1)]
        + [(i, args.width) for i in range(args.width + 1)]
    )
    goals.remove((0, 0))

    problem_specs = set()

    problems = []

    problem_id = 0
    total_num_problems = args.n_train + args.n_valid + args.n_test

    with tqdm(total=total_num_problems) as pbar:
        prefix = "tr_"
        while len(problem_specs) < total_num_problems:
            if len(problem_specs) == args.n_train:
                prefix = "te_"
                problem_id = 0
            head_at_goal = False
            goal = random.choice(goals)
            problem = {
                "init": (0, 0),  # todo allow other initial pos?
                "goal": goal,
            }
            wit = Witness(
                width=args.width, max_num_colors=args.max_num_colors, **problem
            )
            # generate a path from start to goal
            state = wit.reset()
            while actions := wit.actions_unpruned(state):
                action = random.choice(actions)
                state = wit.result(state, action)
                if wit.is_head_at_goal(state):
                    head_at_goal = True
                    break

            if not head_at_goal:
                continue

            regions = connected_components(wit, state)

            min_num_regions = 2
            if args.width == 3:
                min_num_regions = 2
            if args.width >= 4:
                min_num_regions = 4
            if args.width == 10:
                min_num_regions = 5

            if len(regions) < min_num_regions:
                continue

            # fill regions with colors, only keep sufficiently non empty ones
            colors = random.choices(range(1, args.max_num_colors + 1), k=len(regions))
            unique_colors_used = set()
            colored_cells = []
            non_unit_regions_unique_colors = 0
            for region in regions:
                region_arr = np.array(sorted(region))
                region_mask = np.random.rand(len(region_arr)) < args.bullet_prob
                region_arr = region_arr[region_mask]
                if len(region_arr):
                    color = colors.pop()
                    if len(region_arr) > 1 and color not in unique_colors_used:
                        non_unit_regions_unique_colors += 1
                    unique_colors_used.add(color)
                    colored_cells.extend(
                        [f"{row} {col} {color}" for row, col in region_arr]
                    )

            if non_unit_regions_unique_colors < args.width // 2:
                continue

            problem["colored_cells"] = colored_cells
            problem_str = str(problem)
            if problem_str in problem_specs:
                print("duplicate")
                continue
            else:
                problem_specs.add(problem_str)
                problem["id"] = f"{prefix}{problem_id}"
                problem_id += 1
                problems.append(problem)
                pbar.update()

    assert isinstance(wit, Witness)
    num_actions = wit.num_actions
    in_channels = wit.in_channels
    width = wit.width
    max_num_colors = wit.max_num_colors
    state_t_width = wit.state_width

    problemset_dict_template = {
        "domain_module": "witness",
        "domain_name": "Witness",
        "width": width,
        "num_actions": num_actions,
        "max_num_colors": max_num_colors,
        "in_channels": in_channels,
        "state_t_width": state_t_width,
    }

    for n, suffix in [
        (args.n_train, "train"),
        (args.n_valid, "valid"),
        (args.n_test, "test"),
    ]:
        problemset_dict = deepcopy(problemset_dict_template)
        if n == 0:
            continue
        elif suffix == "train":
            problemset_dict["is_curriculum"] = True
            problemset_dict["bootstrap_problems"] = []
            problemset_dict["permutation_problems"] = []

            train_problems = problems[:n]
            problemset_dict["curriculum_problems"] = train_problems
            problemset_dict["curriculum"] = ["unfiltered"]
            problemset_dict["problems_per_difficulty"] = len(train_problems)
        else:
            problemset_dict["problems"] = problems[:n]
        path = args.output_path / f"{n}-{suffix}.json"
        with path.open("w") as f:
            json.dump(problemset_dict, f)
        problems = problems[n:]


def connected_components(wit, wit_state):
    """
    Compute the connected components of the grid, i.e. the regions separated by the path
    """
    visited = np.zeros((wit.width, wit.width))
    cell_states = [(i, j) for i, j in product(range(wit.width), range(wit.width))]
    regions = []
    while len(cell_states) != 0:
        root = cell_states.pop()
        # If root of new BFS search was already visited, then go to the next state
        if visited[root] == 1:
            continue
        this_region = [root]
        frontier = deque()
        frontier.append(root)
        visited[root] = 1
        while len(frontier) != 0:
            cell_state = frontier.popleft()

            def reachable_neighbors(cell):
                neighbors = []
                row, col = cell
                # move up
                if row + 1 < wit.width and wit_state.h_segs[row + 1, col] == 0:
                    neighbors.append((row + 1, col))
                # move down
                if row > 0 and wit_state.h_segs[row, col] == 0:
                    neighbors.append((row - 1, col))
                # move right
                if col + 1 < wit.width and wit_state.v_segs[row, col + 1] == 0:
                    neighbors.append((row, col + 1))
                # move left
                if col > 0 and wit_state.v_segs[row, col] == 0:
                    neighbors.append((row, col - 1))
                return neighbors

            neighbors = reachable_neighbors(cell_state)
            for neighbor in neighbors:
                if visited[neighbor] == 1:
                    continue
                this_region.append(neighbor)
                frontier.append(neighbor)
                visited[neighbor] = 1
        regions.append(this_region)
    return regions


if __name__ == "__main__":
    main()
