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
from typing import Optional, Type

import torch as to

from domains.domain import Domain, State


class SearchNode:
    def __init__(
        self,
        state: State,
        g_cost: float,
        parent: Optional[SearchNode] = None,
        parent_action: Optional[int] = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.g_cost = g_cost

    def __eq__(self, other):
        """
        Verify if two SearchNodes are identical by verifying the
         state in the nodes.
        """
        return self.state == other.state

    def __lt__(self, other):
        """
        less-than used by the heap
        """
        return self.g_cost < other.g_cost

    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self.state.__hash__()


class Trajectory:
    def __init__(
        self,
        domain: Domain,
        final_node: SearchNode,
        num_expanded: int,
        goal_state_t: Optional[to.Tensor] = None,
        forward: bool = True,
    ):
        """
        Receives a SearchNode representing a solution to the problem.
        Backtracks the path performed by search, collecting state-action pairs along the way.
        """
        self.forward = forward
        self.num_expanded = num_expanded
        self.goal_state_t: Optional[to.Tensor] = (
            goal_state_t.unsqueeze(0) if goal_state_t is not None else None
        )

        action = final_node.parent_action
        node = final_node.parent
        states = []
        actions = []

        while node:
            state_t = domain.state_tensor(node.state)
            states.append(state_t)
            actions.append(action)
            action = node.parent_action
            node = node.parent

        self.states = to.stack(states[::-1])
        self.actions = to.tensor(actions[::-1])
        self.cost_to_gos = to.arange(len(self.states), 0, -1)

    def __len__(self):
        return len(self.states)


class MergedTrajectory:
    def __init__(self, trajs: list[Trajectory]):
        if trajs:
            if trajs[0].goal_state_t is not None:
                self.goal_states = to.cat(tuple(t.goal_state_t for t in trajs))
            else:
                self.goal_states = None

            self.states = to.cat(tuple(t.states for t in trajs))
            self.actions = to.cat(tuple(t.actions for t in trajs))
            self.lengths = to.tensor(tuple(len(t) for t in trajs))

            indices = to.arange(len(trajs))
            self.indices = to.repeat_interleave(indices, self.lengths)
            self.nums_expanded = to.tensor(
                tuple(t.num_expanded for t in trajs), dtype=to.float32
            )
            self.num_trajs = len(self.nums_expanded)
            self.num_states = len(self.states)
            self.forward = trajs[0].forward

            # if shuffle:
            #     self.shuffle()
        else:
            return None

    def __len__(self):
        raise NotImplementedError

    # def shuffle(self):
    #     perm = to.randperm(self.num_states)
    #     self.states = self.states[perm]
    #     self.actions = self.actions[perm]
    #     self.indices = self.indices[perm]


def get_merged_solution(
    dir1_domain: Domain,
    dir1_common: SearchNode,
    dir2_common: SearchNode,
    node_type: Type[SearchNode],
    num_expanded: int,
    goal_state_t: Optional[to.Tensor] = None,
    forward: bool = True,
):
    """
    Returns a new trajectory going from dir1_start to dir2_start, passing through
    merge(dir1_common, dir2_common).
    """
    dir1_node = dir1_common
    parent_dir2_node = dir2_common.parent
    parent_dir2_action = dir2_common.parent_action
    while parent_dir2_node:
        new_dir1_node = node_type(
            state=parent_dir2_node.state,
            parent=dir1_node,
            parent_action=dir1_domain.reverse_action(parent_dir2_action),
            g_cost=dir1_node.g_cost + 1,
        )
        dir1_node = new_dir1_node
        parent_dir2_action = parent_dir2_node.parent_action
        parent_dir2_node = parent_dir2_node.parent

    return Trajectory(dir1_domain, dir1_node, num_expanded, goal_state_t, forward)


def try_make_solution(
    this_domain: Domain,
    node: SearchNode,
    other_domain: Domain,
    num_expanded: int,
) -> Optional[tuple[Trajectory, Trajectory]]:
    """
    Returns a trajectory if state is a solution to this problem, None otherwise.
    """
    hsh = node.state.__hash__()
    if hsh in other_domain.visited:  # solution found
        other_node = other_domain.visited[hsh]
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

        f_traj = get_merged_solution(
            f_domain, f_common_node, b_common_node, type(node), num_expanded
        )
        b_traj = get_merged_solution(
            b_domain,
            b_common_node,
            f_common_node,
            type(node),
            num_expanded,
            b_domain.goal_state_t,
            forward=False,
        )

        return (f_traj, b_traj)
    else:
        return None
