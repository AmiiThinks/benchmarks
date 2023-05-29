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

import torch as to
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts

from models import AgentModel
from search.utils import MergedTrajectory


def levin_loss(trajs: MergedTrajectory, model: AgentModel):
    state_feats = model.feature_net(trajs.states)

    if trajs.forward:
        logits = model.forward_policy(state_feats)
    else:
        if trajs.goal_states is not None:
            goal_feats = model.feature_net(trajs.goal_states)
            goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
            assert goal_feats_expanded.shape[0] == state_feats.shape[0]
            logits = model.backward_policy(state_feats, goal_feats_expanded)
        else:
            logits = model.backward_policy(state_feats)

    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    loss = to.dot(traj_nlls, trajs.nums_expanded) / trajs.num_trajs
    avg_action_nll = action_nlls.detach().mean().item()

    return loss, avg_action_nll, logits.detach()


def ub_loss(trajs: MergedTrajectory, model: AgentModel):
    pass
    # state_feats = model.feature_net(trajs.states)

    # if trajs.forward:
    #     logits = model.forward_policy(state_feats)
    # else:
    #     if trajs.goal_states is not None:
    #         goal_feats = model.feature_net(trajs.goal_states)
    #         goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
    #         assert goal_feats_expanded.shape[0] == state_feats.shape[0]
    #         logits = model.backward_policy(state_feats, goal_feats_expanded)
    #     else:
    #         logits = model.backward_policy(state_feats)

    # action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    # traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    # probs = to.exp(-1 * traj_nlls.detach())
    # upperbounds = to.div(trajs.lengths + 1, probs)
    # loss = to.dot(traj_nlls, upperbounds) / trajs.num_trajs
    # avg_action_nll = action_nlls.detach().mean().item()

    # return loss, avg_action_nll, logits.detach()


def min_num_actions_ub_loss(trajs: MergedTrajectory, model: AgentModel):
    pass
    # state_feats = model.feature_net(trajs.states)

    # if trajs.forward:
    #     logits = model.forward_policy(state_feats)
    # else:
    #     if trajs.goal_states is not None:
    #         goal_feats = model.feature_net(trajs.goal_states)
    #         goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
    #         assert goal_feats_expanded.shape[0] == state_feats.shape[0]
    #         logits = model.backward_policy(state_feats, goal_feats_expanded)
    #     else:
    #         logits = model.backward_policy(state_feats)

    # action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    # traj_nlls = ts.scatter_add(action_nlls, trajs.indices, dim=0)
    # upperbounds = to.div(trajs.lengths, trajs.probs)
    # loss = to.dot(traj_nlls, upperbounds) / trajs.num_trajs
    # avg_action_nll = action_nlls.detach().mean().item()

    # return loss, avg_action_nll, logits.detach()


def cross_entropy_loss(trajs: MergedTrajectory, model: AgentModel):
    state_feats = model.feature_net(trajs.states)

    if trajs.forward:
        logits = model.forward_policy(state_feats)
    else:
        if trajs.goal_states is not None:
            goal_feats = model.feature_net(trajs.goal_states)
            goal_feats_expanded = to.repeat_interleave(goal_feats, trajs.lengths, dim=0)
            assert goal_feats_expanded.shape[0] == state_feats.shape[0]
            logits = model.backward_policy(state_feats, goal_feats_expanded)
        else:
            logits = model.backward_policy(state_feats)

    action_nlls = F.cross_entropy(logits, trajs.actions, reduction="none")
    loss = action_nlls.sum() / trajs.num_trajs
    avg_action_nll = action_nlls.detach().mean().item()

    return loss, avg_action_nll, logits.detach()


"""
Originally from https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
"""


def masked_log_softmax(vector: to.Tensor, mask: to.Tensor, dim: int = -1) -> to.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        vector = vector + (mask + 1e-45).log()
    return to.nn.functional.log_softmax(vector, dim=dim)
