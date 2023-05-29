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
from pathlib import Path
import json
import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with problem instances to read (old spec)",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=lambda p: Path(p).absolute(),
        help="path of file to write problem instances (new spec)",
    )

    args = parser.parse_args()

    problem_specs_old = [
        line_list
        for lines in args.input_path.read_text().split("\n\n")
        if len(line_list := lines.splitlines()) == 4
    ]

    problemset = {
        "domain_name": "Witness",
        "domain_module": "witness",
        "max_num_colors": 4,
        "width": 4,
        "problems": [],
    }

    for i, old_spec in tqdm.tqdm(enumerate(problem_specs_old)):
        new_spec = {}

        init = old_spec[1].replace("Init: ", "").split(" ")
        goal = old_spec[2].replace("Goal: ", "").split(" ")
        new_spec = {
            "init": [int(init[0]), int(init[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "id": i,
        }
        colored_cells = []
        values = old_spec[3].replace("Colors: |", "").split("|")
        for t in values:
            if not t:
                break
            numbers = t.split(" ")
            colored_cells.append(
                f"{int(numbers[0])} {int(numbers[1])} {int(numbers[2])}"
            )
        new_spec["colored_cells"] = colored_cells

        problemset["problems"].append(new_spec)

    with args.output_path.open("w") as f:
        json.dump(problemset, f, indent=2)


if __name__ == "__main__":
    main()
