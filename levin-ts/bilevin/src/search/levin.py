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
import heapq
import math
import time
from typing import Optional, TYPE_CHECKING

import torch as to
from torch.jit import RecursiveScriptModule
from torch.nn.functional import log_softmax

from domains.domain import State
from models.loss_functions import masked_log_softmax
from search.agent import Agent
from search.utils import SearchNode, Trajectory

if TYPE_CHECKING:
    from domains.domain import State, Domain, Problem


class Levin(Agent):
    @property
    def bidirectional(cls):
        return False

    @property
    def trainable(cls):
        return True

    def search(
        self,
        problem: Problem,
        budget,
        update_levin_costs=False,
        train=False,
        end_time=None,
    ):
        """ """

        problem_id = problem.id
        domain = problem.domain
        model = self.model
        num_actions = domain.num_actions
        feature_net: RecursiveScriptModule = model.feature_net
        forward_policy: RecursiveScriptModule = model.forward_policy

        state = domain.reset()
        assert isinstance(state, State)
        state_t = domain.state_tensor(state).unsqueeze(0)

        state_feats = feature_net(state_t)
        action_logits = forward_policy(state_feats)

        avail_actions = domain.actions_unpruned(state)
        mask = to.zeros(num_actions)
        mask[avail_actions] = 1
        log_action_probs = masked_log_softmax(action_logits[0], mask, dim=-1)

        node = LevinNode(
            state,
            g_cost=0,
            log_prob=0.0,
            levin_cost=0.0,
            actions=avail_actions,
            log_action_probs=log_action_probs,
        )

        frontier = PriorityQueue()
        reached = {}
        if avail_actions:
            frontier.enqueue(node)
        reached[node] = node

        num_expanded = 0
        num_generated = 0
        while len(frontier) > 0:
            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return 0, num_expanded, num_generated, None

            node = frontier.dequeue()
            num_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = domain.result(node.state, a)
                new_state_actions = domain.actions(a, new_state)

                new_node = LevinNode(
                    new_state,
                    g_cost=node.g_cost + 1,
                    parent=node,
                    parent_action=a,
                    actions=new_state_actions,
                    log_prob=node.log_prob + node.log_action_probs[a].item(),
                )
                new_node.levin_cost = levin_cost(new_node)

                num_generated += 1

                if new_node not in reached:
                    if domain.is_goal(new_state):
                        solution_len = new_node.g_cost
                        traj = Trajectory(domain, new_node, num_expanded)
                        if train:
                            traj = (traj,)
                        return solution_len, num_expanded, num_generated, traj

                    reached[new_node] = new_node
                    if new_state_actions:
                        frontier.enqueue(new_node)

                        children_to_be_evaluated.append(new_node)
                        state_t = domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)

                        mask = to.zeros(num_actions)
                        mask[new_state_actions] = 1
                        masks.append(mask)

                elif update_levin_costs:
                    old_node = reached[new_node]
                    if new_node.g_cost < old_node.g_cost:
                        swap_node_contents(new_node, old_node)
                        if old_node in frontier:
                            frontier.remove(old_node)
                            frontier.enqueue(old_node)

            if children_to_be_evaluated:
                batch_states = to.stack(state_t_of_children_to_be_evaluated)
                batch_feats = feature_net(batch_states)
                action_logits = forward_policy(batch_feats)
                masks = to.stack(masks)
                log_action_probs = masked_log_softmax(action_logits, masks, dim=-1)

                for child, lap in zip(children_to_be_evaluated, log_action_probs):
                    child.log_action_probs = lap

        print(f"Emptied frontier for problem {problem_id}")
        return 0, num_expanded, num_generated, None


def swap_node_contents(src: LevinNode, dst: LevinNode):
    dst.g_cost = src.g_cost
    dst.parent = src.parent
    dst.parent_action = src.parent_action
    dst.log_prob = src.log_prob
    dst.levin_cost = levin_cost(dst)


class PQEntry:
    def __init__(self, node):
        self.node = node
        self.removed = False

    def __lt__(self, other):
        return self.node < other.node


class PriorityQueue:
    def __init__(self) -> None:
        self.pq = []
        self.entry_finder = {}

    def top(self):
        for entry in self.pq:
            if not entry.removed:
                return entry.node
        raise KeyError("top from an empty priority queue")

    def enqueue(self, node):
        if node in self.entry_finder:
            self.remove(node)
        entry = PQEntry(node)
        heapq.heappush(self.pq, entry)
        self.entry_finder[node] = entry

    def dequeue(self):
        while self.pq:
            entry = heapq.heappop(self.pq)
            if not entry.removed:
                del self.entry_finder[entry.node]
                return entry.node
        raise KeyError("pop from an empty priority queue")

    def remove(self, node):
        entry = self.entry_finder.pop(node)
        entry.removed = True

    def __contains__(self, node):
        return node in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)


class LevinNode(SearchNode):
    def __init__(
        self,
        state: State,
        g_cost: float,
        parent: Optional[SearchNode] = None,
        parent_action: Optional[int] = None,
        log_prob: Optional[float] = None,
        levin_cost: Optional[float] = None,
        actions: Optional[list[int]] = None,
        log_action_probs: Optional[to.Tensor] = None,
    ):
        super().__init__(
            state=state, parent=parent, parent_action=parent_action, g_cost=g_cost
        )
        self.log_prob = log_prob
        self.levin_cost = levin_cost
        self.actions = actions
        self.log_action_probs = log_action_probs

    def __lt__(self, other):
        """
        used by the heap
        """
        return self.levin_cost < other.levin_cost


def levin_cost(node: LevinNode):
    return math.log(node.g_cost + 1) - node.log_prob  # type:ignore
