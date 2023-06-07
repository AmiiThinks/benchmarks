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
import json
from pathlib import Path

from copy import copy
import numpy as np
import tqdm

import domains
from domains import SlidingTilePuzzle
from src.train import train


def is_valid(tiles):
    """
    Check if a sliding tile puzzle is solvable.
    For nxn grids.
    if n is odd, then the number of inversions must be even.
    if n is even, then the number of inversions + blank_row (0-indexed) must be even.
    """
    blank_row = np.where(tiles == 0)[0].item()
    num_inversions = 0
    width = tiles.shape[0]
    n = width**2
    tiles = tiles.reshape(-1)
    for i in range(0, n):
        for j in range(i + 1, n):
            if tiles[i] and tiles[j] and tiles[i] > tiles[j]:
                num_inversions += 1

    if width % 2 == 1:
        return num_inversions % 2 == 0
    else:
        return (blank_row + num_inversions) % 2 == 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )
    parser.add_argument(
        "-x",
        "--exclude-problemset",
        action="extend",
        nargs="+",
        type=lambda p: Path(p).absolute(),
        help="path to save problem instances",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=4,
        help="width of puzzles to be generated",
    )
    parser.add_argument(
        "--bootstrap-steps",
        type=int,
        default=10,
        help="generate all problems up to (inclusive) this many steps away from the goal",
    )
    parser.add_argument(
        "--curriculum",
        nargs="+",
        default=[],
        help="list of steps away from goal for the curriculum",
    )
    parser.add_argument(
        "--n-problems-per-difficulty",
        type=int,
        default=3200,
        help="number of training puzzles to be generated",
    )
    parser.add_argument(
        "--n-permutation-problems",
        type=int,
        default=3200,
        help="number of validation puzzles to be generated",
    )
    parser.add_argument(
        "--n-valid",
        type=int,
        default=3200,
        help="number of validation puzzles to be generated",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=3200,
        help="number of testing puzzles to be generated",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    exclude_problemspecs = set()
    if args.exclude_problemset:
        for pset_path in args.exclude_problemset:
            problemset_dict = json.load(pset_path.open("r"))
            domain_module = getattr(domains, problemset_dict["domain_module"])
            problemset, _ = getattr(domain_module, "parse_problemset")(problemset_dict)

            for problem in problemset["problems"]:
                exclude_problemspecs.add(problem.domain.initial_state)

    rng = np.random.default_rng(args.seed)
    goal_tiles = np.arange(args.width**2).reshape(args.width, args.width)
    stp = SlidingTilePuzzle(goal_tiles, goal_tiles)

    problemset_dict = {
        "domain_module": "stp",
        "domain_name": "SlidingTilePuzzle",
        "state_t_width": args.width,
        "width": args.width,
        "num_actions": 4,
        "in_channels": int(args.width**2),
    }

    def generate_permutation_problems(id_prefix, id_start, num_problems, desc):
        problems = []
        id_counter = id_start
        with tqdm.tqdm(total=num_problems) as local_pbar:
            local_pbar.set_description(desc)
            generated = 0
            while generated < num_problems:
                tiles = rng.permutation(args.width**2).reshape(
                    (args.width, args.width)
                )
                stp = SlidingTilePuzzle(tiles, goal_tiles)
                state = stp.initial_state
                if (
                    state in exclude_problemspecs
                    or stp.is_goal(state)
                    or not is_valid(state.tiles)
                ):
                    continue
                else:
                    exclude_problemspecs.add(state)
                    problem = {
                        "tiles": state.tiles.tolist(),
                        "id": f"{id_prefix}_{id_counter}",
                    }
                    problems.append(problem)
                    id_counter += 1
                    generated += 1
                    local_pbar.update(1)
        return problems

    def get_all_reachable_states(
        id_prefix, id_start, state, max_step, exclude_problemspecs
    ):
        problems = []
        id_counter = id_start

        def helper(state, step, max_step):
            nonlocal id_counter
            if step > max_step:
                return
            stp = SlidingTilePuzzle(state.tiles, goal_tiles)
            actions = stp.actions_unpruned(state)
            for action in actions:
                new_state = stp.result(state, action)
                if new_state in exclude_problemspecs or stp.is_goal(new_state):
                    continue
                else:
                    exclude_problemspecs.add(new_state)
                    problem = {
                        "tiles": new_state.tiles.tolist(),
                        "id": f"{id_prefix}_{id_counter}",
                    }
                    id_counter += 1
                    problems.append(problem)
                    helper(new_state, step + 1, max_step)

        helper(state, 1, max_step)
        return problems

    def save_problemset(problemset_dict, suffix, is_curriculum=False):
        n_problems = 0
        if "problems" in problemset_dict:
            n_problems += len(problemset_dict["problems"])
        if "bootstrap_problems" in problemset_dict:
            n_problems += len(problemset_dict["bootstrap_problems"])
        if "curriculum_problems" in problemset_dict:
            n_problems += len(problemset_dict["curriculum_problems"])
        if "permutation_problems" in problemset_dict:
            n_problems += len(problemset_dict["permutation_problems"])

        path = args.output_path / f"{n_problems}-{suffix}.json"
        with path.open("w") as f:
            json.dump(problemset_dict, f)
        print(f"Saved {n_problems} problems to {path}")

    print(
        f"Generating bootstrap problems up to {args.bootstrap_steps} steps away from goal.."
    )
    bootstrap_problems = get_all_reachable_states(
        "b", 0, stp.initial_state, args.bootstrap_steps, exclude_problemspecs
    )
    print(f"Generated {len(bootstrap_problems)} problems.")

    print(
        f"Generating {args.n_problems_per_difficulty} curriculum problems for each of {len(args.curriculum)} steps: {args.curriculum}"
    )
    curriculum_problems = []
    id_counter = 0
    num_curriculum_problems = args.n_problems_per_difficulty * len(args.curriculum)
    with tqdm.tqdm(total=num_curriculum_problems) as pbar:
        pbar.set_description("Curriculum problems")
        difficulty = 0
        steps = int(args.curriculum[difficulty])
        while len(curriculum_problems) < num_curriculum_problems:
            state = stp.reset()
            for _ in range(steps):
                actions = stp.actions_unpruned(state)
                random_action = rng.choice(actions)
                state = stp.result(state, random_action)

            if state in exclude_problemspecs or stp.is_goal(state):
                continue
            else:
                exclude_problemspecs.add(state)
                problem = {
                    "tiles": state.tiles.tolist(),
                    "id": f"c{difficulty}_{id_counter}",
                }
                curriculum_problems.append(problem)
                id_counter += 1
                pbar.update(1)
                if (
                    len(curriculum_problems) < num_curriculum_problems
                    and len(curriculum_problems) % args.n_problems_per_difficulty == 0
                ):
                    difficulty += 1
                    id_counter = 0
                    steps = int(args.curriculum[difficulty])

    permutation_problems = generate_permutation_problems(
        "p",
        0,
        args.n_permutation_problems,
        "Permutation problems",
    )

    trainset_dict = copy(problemset_dict)
    trainset_dict["is_curriculum"] = True
    trainset_dict["curriculum"] = args.curriculum
    trainset_dict["problems_per_difficulty"] = args.n_problems_per_difficulty
    trainset_dict["bootstrap_problems"] = bootstrap_problems
    trainset_dict["curriculum_problems"] = curriculum_problems
    trainset_dict["permutation_problems"] = permutation_problems
    save_problemset(trainset_dict, "train")

    if args.n_valid > 0:
        valid_problems = generate_permutation_problems(
            "v", 0, args.n_valid, "Validation problems"
        )
        problemset_dict["problems"] = valid_problems
        save_problemset(
            problemset_dict,
            "valid",
        )

    if args.n_test > 0:
        test_problems = generate_permutation_problems(
            "t", 0, args.n_test, "Test problems"
        )
        problemset_dict["problems"] = test_problems
        save_problemset(
            problemset_dict,
            "test",
        )


if __name__ == "__main__":
    main()
