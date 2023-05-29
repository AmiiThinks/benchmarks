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
from copy import deepcopy
import time
from typing import TYPE_CHECKING

import torch as to
from torch.jit import RecursiveScriptModule
from torch.nn.functional import log_softmax

from domains.domain import State
from enums import TwoDir
from models.loss_functions import masked_log_softmax
from search.agent import Agent
from search.levin import LevinNode, PriorityQueue, levin_cost, swap_node_contents

if TYPE_CHECKING:
    from domains.domain import Domain, Problem


class BiLevin(Agent):
    @property
    def bidirectional(cls):
        return True

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
        f_frontier = PriorityQueue()
        b_frontier = PriorityQueue()
        f_reached = {}
        b_reached = {}

        model = self.model
        feature_net: RecursiveScriptModule = model.feature_net
        forward_policy: RecursiveScriptModule = model.forward_policy
        backward_policy: RecursiveScriptModule = model.backward_policy
        double_backward = model.double_backward

        problem_id = problem.id
        f_domain = problem.domain
        num_actions = f_domain.num_actions

        try_make_solution = f_domain.try_make_solution_func

        b_domain = f_domain.backward_domain()

        f_state = f_domain.reset()
        assert isinstance(f_state, State)
        f_state_t = f_domain.state_tensor(f_state).unsqueeze(0)

        f_avail_actions = f_domain.actions_unpruned(f_state)
        f_mask = to.zeros(num_actions)
        f_mask[f_avail_actions] = 1

        b_states = b_domain.reset()
        if isinstance(b_states, list):
            b_state_t = []
            b_avail_actions = []
            b_mask = to.zeros((len(b_states), num_actions))
            for i, s in enumerate(b_states):
                b_state_t.append(b_domain.state_tensor(s))
                b_avail_actions.append(b_domain.actions_unpruned(s))
                b_mask[i, b_avail_actions[-1]] = 1
            b_state_t = to.stack(b_state_t)
        else:
            b_state_t = b_domain.state_tensor(b_states)
            b_state_t = b_state_t.unsqueeze(0)
            b_avail_actions = b_domain.actions_unpruned(b_states)
            b_mask = to.zeros(num_actions)
            b_mask[b_avail_actions] = 1
            b_states = [b_states]
            b_avail_actions = [b_avail_actions]

        feats = feature_net(to.vstack((f_state_t, b_state_t)))
        f_state_feats = feats[0]
        b_states_feat = feats[1:]

        b_goal_feats = deepcopy(f_state_feats)

        f_action_logits = forward_policy(f_state_feats)

        if double_backward:
            b_action_logits = backward_policy(b_states_feat, b_goal_feats)
        else:
            b_action_logits = backward_policy(b_states_feat)

        f_log_action_probs = masked_log_softmax(f_action_logits, f_mask, dim=-1)
        b_log_action_probs = masked_log_softmax(b_action_logits, b_mask, dim=-1)

        f_start_node = LevinNode(
            f_state,
            g_cost=0,
            log_prob=0.0,
            levin_cost=0.0,
            actions=f_avail_actions,
            log_action_probs=f_log_action_probs,
        )
        f_reached[f_start_node] = f_start_node
        f_domain.update(f_start_node)
        if f_avail_actions:
            f_frontier.enqueue(f_start_node)

        for i, state in enumerate(b_states):
            start_node = LevinNode(
                state,
                g_cost=0,
                log_prob=0.0,
                levin_cost=0.0,
                actions=b_avail_actions[i],
                log_action_probs=b_log_action_probs[i],
            )
            b_reached[start_node] = start_node
            b_domain.update(start_node)
            if start_node.actions:
                b_frontier.enqueue(start_node)

        num_expanded = 0
        num_generated = 0
        while len(f_frontier) > 0 and len(b_frontier) > 0:
            if (
                (budget and num_expanded >= budget)
                or end_time
                and time.time() > end_time
            ):
                return (False, num_expanded, num_generated, None)

            b = b_frontier.top()
            f = f_frontier.top()

            if f < b:
                direction = TwoDir.FORWARD
                _domain = f_domain
                _policy = forward_policy
                _frontier = f_frontier
                _reached = f_reached
                other_domain = b_domain
            else:
                direction = TwoDir.BACKWARD
                _domain = b_domain
                _policy = backward_policy
                _frontier = b_frontier
                _reached = b_reached
                other_domain = f_domain

            node = _frontier.dequeue()
            num_expanded += 1

            masks = []
            children_to_be_evaluated = []
            state_t_of_children_to_be_evaluated = []
            for a in node.actions:
                new_state = _domain.result(node.state, a)
                new_state_actions = _domain.actions(a, new_state)

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

                if new_node not in _reached:
                    trajs = try_make_solution(
                        _domain, new_node, other_domain, num_expanded
                    )

                    if trajs:  # solution found
                        solution_len = len(trajs[0])
                        assert solution_len == len(trajs[1])
                        if not train:
                            trajs = trajs[0]
                        return solution_len, num_expanded, num_generated, trajs

                    _reached[new_node] = new_node
                    _domain.update(new_node)

                    if new_state_actions:
                        _frontier.enqueue(new_node)
                        children_to_be_evaluated.append(new_node)
                        state_t = _domain.state_tensor(new_state)
                        state_t_of_children_to_be_evaluated.append(state_t)

                        mask = to.zeros(num_actions)
                        mask[new_state_actions] = 1
                        masks.append(mask)

                elif update_levin_costs:
                    old_node = _reached[new_node]
                    if new_node.g_cost < old_node.g_cost:
                        # print("updating")
                        swap_node_contents(new_node, old_node)
                        if old_node in _frontier:
                            # print("updating frontier")
                            _frontier.remove(old_node)
                            _frontier.enqueue(old_node)

            if children_to_be_evaluated:
                batch_states = to.stack(state_t_of_children_to_be_evaluated)
                batch_feats = feature_net(batch_states)
                if direction == TwoDir.BACKWARD:
                    if double_backward:
                        action_logits = _policy(batch_feats, b_goal_feats)
                    else:
                        action_logits = _policy(batch_feats)
                else:
                    action_logits = _policy(batch_feats)

                masks = to.stack(masks)
                log_action_probs = masked_log_softmax(action_logits, masks, dim=-1)

                for i, child in enumerate(children_to_be_evaluated):
                    child.log_action_probs = log_action_probs[i]

        print(f"Emptied frontiers for problem {problem_id}")
        return False, num_expanded, num_generated, None
