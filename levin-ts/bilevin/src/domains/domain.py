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
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, TYPE_CHECKING, Callable

import torch as to

if TYPE_CHECKING:
    from search.utils import SearchNode, Trajectory


class Problem:
    def __init__(self, id: str, domain: Domain):
        self.id = id
        self.domain = domain

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.id == other.id


class State(ABC):
    @abstractmethod
    def __eq__(self) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


class Domain(ABC):
    def __init__(self, forward: bool = True):
        self.visited: dict = {}
        self.forward = forward
        self.initial_state: State | list[State]

    def reset(self) -> State | list[State]:
        self.visited = {}
        return self.initial_state

    def update(self, node: SearchNode):
        self.visited[node.state.__hash__()] = node

    @property
    @abstractmethod
    def try_make_solution_func(
        cls,
    ) -> Callable[
        [Domain, SearchNode, Domain, int], Optional[tuple[Trajectory, Trajectory]]
    ]:
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def state_width(self) -> int:
        pass

    @abstractmethod
    def is_goal(self, state: State) -> bool:
        pass

    @abstractmethod
    def backward_domain(self) -> Domain:
        pass

    @abstractmethod
    def reverse_action(self, action) -> int:
        pass

    @abstractmethod
    def state_tensor(self, state: State) -> to.Tensor:
        pass

    @abstractmethod
    def actions(self, action, state: State) -> list:
        pass

    @abstractmethod
    def actions_unpruned(self, state: State) -> list:
        pass

    @abstractmethod
    def result(self, state: State, action) -> State:
        pass
