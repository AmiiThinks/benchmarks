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
import pathlib
import pickle

import numpy as np
import tqdm

from domains.sokoban import Sokoban, SokobanState


def from_string(
    string_state: str,
) -> Sokoban:
    lines = string_state.splitlines()
    rows = len(lines)
    cols = len(lines[0])

    map = np.zeros((2, rows, cols), dtype=np.float32)
    boxes = np.zeros((rows, cols), dtype=np.int8)
    man_row = -1
    man_col = -1

    for i in range(rows):
        for j in range(cols):
            if lines[i][j] == Sokoban.goal_str:
                map[Sokoban.goal_channel, i, j] = 1

            if lines[i][j] == Sokoban.man_str:
                man_row = i
                man_col = j

            if lines[i][j] == Sokoban.wall_str:
                map[Sokoban.wall_channel, i, j] = 1

            if lines[i][j] == Sokoban.box_str:
                boxes[i, j] = 1

    return Sokoban(map, man_row, man_col, boxes)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=lambda p: Path(p).absolute(),
        help="path of directory contianing problem files, each with instances, to read (old spec)",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="",
        help="prefix to add to problem id ([prefix]_id)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="save as a test problemset (default is train)",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path of file to write problem instances (new spec)",
    )

    args = parser.parse_args()
    if args.id_prefix:
        id_prefix = f"{args.id_prefix}_"
    else:
        id_prefix = ""

    problem_files = sorted(pathlib.Path(args.input_path).glob("*.txt"))
    problemset = []
    for f in tqdm.tqdm(problem_files):
        all_txt = f.read_text()
        all_problem_strings = all_txt.split("\n\n")
        for problem_string in all_problem_strings:
            if not problem_string:
                continue
            num, new_line, problem_string = problem_string.partition("\n")
            problem_id = f"{id_prefix}{f.stem}_{int(num[1:])}"
            problem = from_string(problem_string)
            new_spec = {
                "map": problem.map.tolist(),
                "man_row": problem.initial_state.man_row,
                "man_col": problem.initial_state.man_col,
                "boxes": problem.original_boxes.tolist(),
                "id": problem_id,
            }
            problemset.append(new_spec)

    width = problem.cols
    in_channels = problem.in_channels

    # todo these aren't really a curriculum
    problemset_dict = {
        "domain_name": "Sokoban",
        "domain_module": "sokoban",
        "num_actions": 4,
        "in_channels": in_channels,
        "state_t_width": width,
    }
    if args.test:
        problemset_dict["problems"] = problemset
    else:
        problemset_dict["is_curriculum"] = True
        problemset_dict["bootstrap_problems"] = []
        problemset_dict["permutation_problems"] = []

        problemset_dict["curriculum_problems"] = problemset
        problemset_dict["curriculum"] = ["unfiltered"]
        problemset_dict["problems_per_difficulty"] = len(problemset)

    with args.output_path.open("w") as f:
        json.dump(problemset_dict, f, indent=0)


if __name__ == "__main__":
    main()
