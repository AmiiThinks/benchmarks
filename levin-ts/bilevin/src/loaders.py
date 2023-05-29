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

from copy import copy
from math import ceil

import numpy as np

from domains.domain import Problem


class ProblemsBatchLoader:
    def __init__(
        self,
        problems: list[Problem],
        all_ids: list[str],
        local_batch_size: int,
        world_size: int = 1,
        epochs: int = 1,
        shuffle: bool = True,
        rng=None,
        seed: int = 1,
    ):
        if not rng:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        self.shuffle = shuffle
        self.epochs = epochs
        self.problems = np.empty(len(problems), dtype=object)
        self.problems[:] = problems
        self.all_ids = all_ids  # ids of problems accross all ranks
        self.local_batch_size = local_batch_size
        self._len = len(problems)
        self._num_problems_served = 0

        self.world_size = world_size
        self.world_num_batches = ceil(len(all_ids) / (local_batch_size * world_size))
        self.batches_served = 0

    def __len__(self):
        return self._len

    def __iter__(self):
        self.batches_served = 0
        if self.shuffle:
            self._indices = self.rng.permutation(self._len)
        else:
            self._indices = np.arange(self._len)

        self._num_problems_served = 0

        return self

    def __next__(self):
        if self._num_problems_served >= self._len:
            if self.batches_served < self.world_num_batches:
                self.batches_served += 1
                return []
            raise StopIteration
        next_indices = self._indices[
            self._num_problems_served : self._num_problems_served
            + self.local_batch_size
        ]
        self._num_problems_served += len(next_indices)
        self.batches_served += 1
        return self.problems[next_indices]

    def __getitem__(self, idx):
        return self.problems[idx]


class CurriculumLoader:
    def __init__(
        self,
        local_bootstrap_problems: list[Problem],
        world_bootstrap_ids: list[str],
        bootstrap_epochs: int,
        curriculum: list[int],
        world_problems_per_difficulty: int,
        local_curriculum_problems: list[list[Problem]],
        world_curriculum_ids: list[list[str]],
        curriculum_epochs: int,
        local_permutation_problems: list[Problem],
        world_permutation_ids: list[str],
        permutation_epochs: int,
        local_batch_size: int,
        world_size: int,
        include_prev_difficulty: bool,
        permutation_focus: bool,
        seed: int = 1,
        shuffle: bool = True,
    ):
        self.shuffle = shuffle
        self.local_batch_size = local_batch_size
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.all_bootstrap_ids = world_bootstrap_ids

        self.bootstrap_problems = local_bootstrap_problems
        self.bootstrap_epochs = bootstrap_epochs

        self.curriculum = curriculum
        self.num_curriculum_stages = len(curriculum)
        self.local_curriculum_problems = local_curriculum_problems
        self.world_curriculum_ids = world_curriculum_ids
        self.world_problems_per_difficulty = world_problems_per_difficulty
        self.local_problems_per_difficulty = len(local_curriculum_problems[0])
        self.curriculum_epochs = curriculum_epochs

        self.permutation_problems = local_permutation_problems
        self.all_permutation_ids = world_permutation_ids
        self.permutation_epochs = permutation_epochs
        self.world_size = world_size

        self.include_prev_difficulty = include_prev_difficulty
        self.permutation_focus = permutation_focus

    def __iter__(self):
        self.next_stage = "bootstrap"
        return self

    def __next__(self):
        if self.next_stage == "bootstrap":
            self.next_stage = "curriculum"
            self.curriculum_stage = -1
            self.stage = "bootstrap"
            self.problems = copy(self.bootstrap_problems)
            self.ids = copy(self.all_bootstrap_ids)
            self.loader = ProblemsBatchLoader(
                self.problems,
                self.ids,
                self.local_batch_size,
                self.world_size,
                self.bootstrap_epochs,
                False,  # don't shuffle the bootstrap loader on first epoch
                self.rng,
            )
        elif "curriculum" in self.next_stage:
            self.curriculum_stage += 1
            if self.curriculum_stage == self.num_curriculum_stages - 1:
                self.next_stage = "permutation"
            self.stage = f"curriculum_{self.curriculum_stage}"
            new_problems = self.local_curriculum_problems[self.curriculum_stage]
            new_ids = self.world_curriculum_ids[self.curriculum_stage]
            if self.include_prev_difficulty:
                self.problems.extend(new_problems)
                self.ids.extend(new_ids)
            else:
                self.problems = new_problems
                self.ids = new_ids

            self.loader = ProblemsBatchLoader(
                self.problems,
                self.ids,
                self.local_batch_size,
                self.world_size,
                self.curriculum_epochs,
                self.shuffle,
                self.rng,
            )
        elif self.next_stage == "permutation":
            self.next_stage = "end"
            self.stage = "permutation"
            if self.include_prev_difficulty and not self.permutation_focus:
                self.problems.extend(self.permutation_problems)
                self.ids.extend(self.all_permutation_ids)
            else:
                self.problems = self.permutation_problems
                self.ids = self.all_permutation_ids
            self.loader = ProblemsBatchLoader(
                self.problems,
                self.ids,
                self.local_batch_size,
                self.world_size,
                self.permutation_epochs,
                self.shuffle,
                self.rng,
            )
        elif self.next_stage == "end":
            raise StopIteration

        return self.loader
