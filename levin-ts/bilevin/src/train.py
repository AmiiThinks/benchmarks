# offlini Copyright (C) 2021-2022, Ken Tjhia
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

import math
from math import isclose
from pathlib import Path
import sys
import time
from typing import Callable, Type, Union
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate
from test import test
import torch as to
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from domains.domain import Problem
from loaders import CurriculumLoader, ProblemsBatchLoader
from models import AgentModel
from search import MergedTrajectory
from search.agent import Agent


def train(
    rank: int,
    agent: Agent,
    train_loader: CurriculumLoader,
    writer: SummaryWriter,
    world_size: int,
    expansion_budget: int,
    time_budget: int,
    seed: int,
    grad_steps: int = 10,
    epoch_reduce_lr: int = 99999,
    epoch_reduce_grad_steps: int = 99999,
    epoch_begin_validate: int = 1,
    valid_loader: Optional[ProblemsBatchLoader] = None,
):
    is_distributed = world_size > 1

    param_log_interval = 20

    search_result_header = [
        "ProblemId",
        "SolutionLength",
        "NumExpanded",
        "NumGenerated",
        "Time",
    ]

    opt_result_header = (
        f"           Forward        Backward\nOptStep   Loss    Acc    Loss    Acc"
    )

    bidirectional = agent.bidirectional
    model = agent.model
    optimizer = agent.optimizer
    loss_fn = agent.loss_fn

    for param in model.parameters():
        if not param.grad:
            param.grad = to.zeros_like(param)

    # if rank == 0:
    #     log_params(writer, model, 0)

    local_batch_opt_results = to.zeros(5, dtype=to.float64)

    local_batch_size = train_loader.local_batch_size
    local_batch_search_results = to.zeros(local_batch_size, 5, dtype=to.int64)
    world_batch_search_results = [
        to.zeros((local_batch_size, 5), dtype=to.int64) for _ in range(world_size)
    ]

    batches_seen = 0
    solved_problems = set()
    total_num_expanded = 0
    opt_step = 1
    opt_passes = 1

    num_valid_problems = 0 if not valid_loader else len(valid_loader.all_ids)
    max_valid_expanded = num_valid_problems * expansion_budget
    best_valid_solved = 0
    best_valid_total_expanded = max_valid_expanded

    epoch = 1
    for batch_loader in train_loader:
        # print(f"{train_loader.stage} rank {rank} global batch len {len(batch_loader.all_ids)}")
        # print(f"{train_loader.stage} {batch_loader.all_ids}")
        # print(f"{train_loader.stage} rank {rank} rank batch len {len(batch_loader)}")
        world_num_problems = len(batch_loader.all_ids)
        if world_num_problems == 0:
            continue
        max_epoch_expansions = world_num_problems * expansion_budget

        world_batches_this_difficulty = math.ceil(
            world_num_problems / (local_batch_size * world_size)
        )

        dummy_data = np.column_stack(
            (
                np.zeros(
                    (world_num_problems, len(search_result_header) - 1),
                    dtype=np.int64,
                ),
            )
        )
        world_results_df = pd.DataFrame(dummy_data, columns=search_result_header[1:])
        del dummy_data
        world_results_df["Time"] = world_results_df["Time"].astype(float, copy=False)
        world_results_df["ProblemId"] = batch_loader.all_ids
        world_results_df.set_index("ProblemId", inplace=True)

        world_epoch_f_loss = np.zeros(world_batches_this_difficulty)
        world_epoch_f_acc = np.zeros(world_batches_this_difficulty)
        world_epoch_b_loss = np.zeros(world_batches_this_difficulty)
        world_epoch_b_acc = np.zeros(world_batches_this_difficulty)

        for stage_epoch in range(1, batch_loader.epochs + 1):
            num_new_problems_solved_this_epoch = 0
            num_problems_solved_this_epoch = 0

            if rank == 0:
                print(
                    "============================================================================"
                )
                print(
                    f"START | STAGE {train_loader.stage} EPOCH {stage_epoch} | TOTAL EPOCH {epoch}"
                )

            if epoch == epoch_reduce_lr:
                new_lr = optimizer.param_groups[0]["lr"] * 0.1
                if rank == 0:
                    print(
                        f"-> Learning rate reduced from {optimizer.param_groups[0]['lr']} to {new_lr}"
                    )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

            if epoch == epoch_reduce_grad_steps:
                old_gs = grad_steps
                grad_steps = grad_steps // 2
                if rank == 0:
                    print(f"-> Grad steps reduced from {old_gs} to {grad_steps}")
            if rank == 0:
                print(
                    "============================================================================\n"
                )
                problems_loader = tqdm.tqdm(
                    batch_loader, total=world_batches_this_difficulty
                )
            else:
                problems_loader = batch_loader

            for batch_idx, local_batch_problems in enumerate(problems_loader):
                batches_seen += 1
                local_batch_search_results[
                    :
                ] = 0  # since final batch might contain <= local_batch_size problems

                if rank == 0:
                    print(f"\n\nBatch {batches_seen}")

                model.eval()
                to.set_grad_enabled(False)

                forward_trajs = []
                backward_trajs = []

                num_problems_solved_this_batch = 0
                for i, problem in enumerate(local_batch_problems):
                    start_time = time.time()
                    (
                        solution_length,
                        num_expanded,
                        num_generated,
                        traj,
                    ) = agent.search(
                        problem,
                        expansion_budget,
                        train=True,
                        end_time=start_time + time_budget,
                    )
                    end_time = time.time()
                    if bidirectional:
                        problem.domain.reset()

                    local_batch_search_results[i, 0] = problem.id_idx
                    local_batch_search_results[i, 1] = solution_length
                    local_batch_search_results[i, 2] = num_expanded
                    local_batch_search_results[i, 3] = num_generated
                    local_batch_search_results[i, 4] = int(
                        (end_time - start_time) * 1000
                    )

                    if traj:
                        forward_trajs.append(traj[0])
                        if bidirectional:
                            backward_trajs.append(traj[1])

                if is_distributed:
                    dist.all_gather(
                        world_batch_search_results, local_batch_search_results
                    )
                    world_batch_results_t = to.cat(world_batch_search_results, dim=0)
                else:
                    world_batch_results_t = local_batch_search_results

                world_batch_results_arr = world_batch_results_t.numpy()
                # hacky way to filter out results from partial batches
                world_batch_results_arr = world_batch_results_arr[
                    world_batch_results_arr[:, 2] > 0
                ]

                world_batch_ids = np.array(
                    [batch_loader.all_ids[i] for i in world_batch_results_arr[:, 0]],
                    dtype=np.unicode_,
                )
                world_results_df.loc[
                    world_batch_ids, search_result_header[1:-1]
                ] = world_batch_results_arr[
                    :, 1:-1
                ]  # ProblemId is already index, Time is set in following lines
                world_results_df.loc[world_batch_ids, "Time"] = (
                    world_batch_results_arr[:, -1].astype(float) / 1000
                )

                world_batch_results_df = world_results_df.loc[world_batch_ids]
                world_batch_results_df.sort_values("NumExpanded", inplace=True)

                batch_solved_ids = world_batch_ids[world_batch_results_arr[:, 1] > 0]
                for problem_id in batch_solved_ids:
                    if problem_id not in solved_problems:
                        num_new_problems_solved_this_epoch += 1
                        solved_problems.add(problem_id)

                num_problems_solved_this_batch = len(batch_solved_ids)
                num_problems_solved_this_epoch += num_problems_solved_this_batch
                num_problems_this_batch = len(world_batch_results_arr)

                batch_expansions = world_batch_results_df["NumExpanded"].sum()
                batch_expansions_ratio = batch_expansions / (
                    len(world_batch_results_df) * expansion_budget
                )

                if rank == 0:
                    print(
                        tabulate(
                            world_batch_results_df,
                            headers="keys",
                            tablefmt="psql",
                        )
                    )
                    print(
                        f"Solved {num_problems_solved_this_batch}/{num_problems_this_batch}\n"
                    )
                    print(f"Expansion ratio: {batch_expansions_ratio}\n")
                    total_num_expanded += world_batch_results_df["NumExpanded"].sum()

                    writer.add_scalar(
                        "cum_unique_solved_vs_expanded",
                        len(solved_problems),
                        total_num_expanded,
                    )

                if rank == 0:
                    print(opt_result_header)

                f_merged_traj = MergedTrajectory(forward_trajs)
                if bidirectional:
                    b_merged_traj = MergedTrajectory(backward_trajs)

                to.set_grad_enabled(True)
                model.train()

                num_procs_found_solution = 0
                f_loss = -1
                f_acc = -1
                b_loss = -1
                b_acc = -1

                for grad_step in range(1, grad_steps + 1):
                    optimizer.zero_grad()
                    if forward_trajs:
                        f_loss, f_avg_action_nll, f_logits = loss_fn(
                            f_merged_traj, model
                        )

                        f_acc = (
                            f_logits.argmax(dim=1) == f_merged_traj.actions
                        ).sum().item() / len(f_logits)

                        local_batch_opt_results[0] = f_avg_action_nll
                        local_batch_opt_results[1] = f_acc
                        local_batch_opt_results[2] = 1

                        if bidirectional:
                            b_loss, b_avg_action_nll, b_logits = loss_fn(
                                b_merged_traj, model
                            )
                            b_acc = (
                                b_logits.argmax(dim=1) == b_merged_traj.actions
                            ).sum().item() / len(b_logits)

                            local_batch_opt_results[3] = b_avg_action_nll
                            local_batch_opt_results[4] = b_acc

                            loss = f_loss + b_loss
                        else:
                            loss = f_loss

                        loss.backward()
                    else:
                        local_batch_opt_results[:] = 0

                    if is_distributed:
                        dist.all_reduce(local_batch_opt_results, op=dist.ReduceOp.SUM)
                        num_procs_found_solution = int(
                            local_batch_opt_results[2].item()
                        )
                        if num_procs_found_solution > 0:
                            sync_grads(model, num_procs_found_solution)
                    else:
                        num_procs_found_solution = int(
                            local_batch_opt_results[2].item()
                        )

                    # todo grad clipping? for now inspect norms
                    # if num_procs_found_solution > 0 and rank == 0:
                    #     log_grad_norm(
                    #         model.feature_net.parameters(),
                    #         "feature_net",
                    #         writer,
                    #         opt_step,
                    #     )
                    #     log_grad_norm(
                    #         model.forward_policy.parameters(),
                    #         "forward",
                    #         writer,
                    #         opt_step,
                    #     )
                    #     if bidirectional:
                    #         log_grad_norm(
                    #             model.backward_policy.parameters(),
                    #             "backward",
                    #             writer,
                    #             opt_step,
                    #         )

                    optimizer.step()

                    if num_procs_found_solution > 0:
                        if rank == 0:
                            if grad_step == 1 or grad_step == grad_steps:
                                f_loss = (
                                    local_batch_opt_results[0].item()
                                    / num_procs_found_solution
                                )
                                f_acc = (
                                    local_batch_opt_results[1].item()
                                    / num_procs_found_solution
                                )
                                b_loss = (
                                    local_batch_opt_results[3].item()
                                    / num_procs_found_solution
                                )
                                b_acc = (
                                    local_batch_opt_results[4].item()
                                    / num_procs_found_solution
                                )
                                if bidirectional:
                                    print(
                                        f"{opt_step:7}  {f_loss:5.3f}  {f_acc:5.3f}    {b_loss:5.3f}  {b_acc:5.3f}"
                                    )
                                else:
                                    print(f"{opt_step:7}  {f_loss:5.3f}  {f_acc:5.3f}")
                                if grad_step == grad_steps:
                                    # fmt: off
                                    writer.add_scalar( f"loss_vs_opt_pass/forward", f_loss, opt_passes,)
                                    writer.add_scalar( f"acc_vs_opt_pass/forward", f_acc, opt_passes,)
                                    if bidirectional:
                                        writer.add_scalar( f"loss_vs_opt_pass/backward", b_loss, opt_passes,)
                                        writer.add_scalar( f"acc_vs_opt_pass/backward", b_acc, opt_passes,)
                                    # fmt:on
                        opt_step += 1
                if num_procs_found_solution > 0:
                    opt_passes += 1

                world_epoch_f_loss[batch_idx] = f_loss
                world_epoch_f_acc[batch_idx] = f_acc
                if bidirectional:
                    world_epoch_b_loss[batch_idx] = b_loss
                    world_epoch_b_acc[batch_idx] = b_acc

                if rank == 0:
                    # if batches_seen % param_log_interval == 0:
                    #     log_params(writer, model, batches_seen)

                    batch_avg = num_problems_solved_this_batch / num_problems_this_batch
                    # fmt: off
                    writer.add_scalar(f"solved_vs_batch", batch_avg, batches_seen)
                    writer.add_scalar(f"expansions_vs_batch", batch_expansions_ratio, batches_seen)
                    # writer.add_scalar(f"cum_unique_solved_vs_batch", len(solved_problems), batches_seen)
                    # fmt: on
                    sys.stdout.flush()

            if rank == 0:
                epoch_expansions = world_results_df["NumExpanded"].sum()
                epoch_expansions_ratio = epoch_expansions / max_epoch_expansions
                epoch_solved_ratio = num_problems_solved_this_epoch / world_num_problems
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    epoch_f_loss = world_epoch_f_loss.mean(
                        where=(world_epoch_f_loss >= 0)
                    )
                    epoch_f_acc = world_epoch_f_acc.mean(where=(world_epoch_f_acc >= 0))
                    if bidirectional:
                        epoch_b_loss = world_epoch_b_loss.mean(
                            where=(world_epoch_b_loss >= 0)
                        )
                        epoch_b_acc = world_epoch_b_acc.mean(
                            where=(world_epoch_b_acc >= 0)
                        )
                print(
                    "============================================================================"
                )
                print(
                    f"END | STAGE {train_loader.stage} EPOCH {stage_epoch} | TOTAL EPOCH {epoch}"
                )
                print(
                    "----------------------------------------------------------------------------"
                )
                print(
                    f"SOLVED {num_problems_solved_this_epoch}/{world_num_problems} {epoch_solved_ratio}\n"
                    f"EXPANSIONS {epoch_expansions}/{max_epoch_expansions}  {epoch_expansions_ratio:5.3f}\n"
                )
                print(f"  Forward        Backward\nLoss    Acc    Loss    Acc")
                if bidirectional:
                    print(
                        f"{epoch_f_loss:5.3f}  {epoch_f_acc:5.3f}    {epoch_b_loss:5.3f}  {epoch_b_acc:5.3f}"
                    )
                else:
                    print(f"{epoch_f_loss:5.3f}  {epoch_f_acc:5.3f}")
                print(
                    "============================================================================"
                )

                # fmt: off
                # writer.add_scalar("budget_vs_epoch", budget, epoch)
                # writer.add_scalar(f"budget_{budget}/solved_vs_epoch", epoch_solved, epoch)
                writer.add_scalar(f"solved_vs_epoch", epoch_solved_ratio, epoch)
                writer.add_scalar("cum_unique_solved_vs_epoch", len(solved_problems), epoch)

                writer.add_scalar(f"loss_vs_epoch/forward", epoch_f_loss, epoch)
                writer.add_scalar(f"acc_vs_epoch/forward", epoch_f_acc, epoch)
                if bidirectional:
                    writer.add_scalar(f"loss_vs_epoch/backward", epoch_b_loss, epoch)
                    writer.add_scalar(f"acc_vs_epoch/backward", epoch_b_acc, epoch)

                # writer.add_scalar(f"expansions_vs_epoch", expansions, epoch)
                writer.add_scalar(f"expansions_ratio_vs_epoch", epoch_expansions_ratio, epoch)

                world_results_df.to_csv(f"{writer.log_dir}/epoch_{epoch}.csv")
                # fmt: on
                sys.stdout.flush()

            if valid_loader and epoch >= epoch_begin_validate:
                # print(f"rank {rank}")
                if rank == 0:
                    print("VALIDATION")
                if is_distributed:
                    dist.barrier()
                valid_results = test(
                    rank,
                    agent,
                    valid_loader,
                    writer,
                    world_size,
                    expansion_budget,
                    increase_budget=False,
                    print_results=False,
                    validate=True,
                    epoch=epoch,
                )

                if rank == 0:
                    valid_solved, valid_total_expanded = valid_results
                    valid_expansions_ratio = valid_total_expanded / max_valid_expanded
                    valid_solve_rate = valid_solved / num_valid_problems
                    print(
                        f"SOLVED:  {valid_solved}/{num_valid_problems} {valid_solve_rate:5.3f}"
                    )
                    print(
                        f"EXPANSIONS: {valid_total_expanded}/{max_valid_expanded} {valid_expansions_ratio:5.3f}"
                    )
                    # writer.add_scalar(f"budget_{budget}/valid_solve_rate", valid_solve_rate, epoch)
                    writer.add_scalar(f"valid_solved_vs_epoch", valid_solve_rate, epoch)
                    writer.add_scalar(
                        f"valid_expansions_ratio_vs_epoch",
                        valid_expansions_ratio,
                        epoch,
                    )
                    # writer.add_scalar( f"valid_expanded_vs_epoch", valid_total_expanded, epoch)

                    agent.save_model("latest", log=False)
                    if valid_total_expanded < best_valid_total_expanded or (
                        isclose(valid_total_expanded, best_valid_total_expanded)
                        and valid_solved > best_valid_solved
                    ):
                        best_valid_total_expanded = valid_total_expanded
                        print("Saving best model by expansions")
                        writer.add_text(
                            "best_model_expansions",
                            f"epoch: {epoch}, solve rate: {valid_solve_rate}, expansion ratio: {valid_expansions_ratio}",
                        )
                        agent.save_model("best_expanded", log=False)

                    if valid_solved > best_valid_solved or (
                        isclose(valid_solved, best_valid_solved)
                        and valid_total_expanded > best_valid_total_expanded
                    ):
                        best_valid_solved = valid_solved
                        print("Saving best model by solved")
                        writer.add_text(
                            "best_model_solved",
                            f"epoch: {epoch}, solve rate: {valid_solve_rate}, expansion ratio: {valid_expansions_ratio}",
                        )
                        agent.save_model("best_solved", log=False)

                if is_distributed:
                    dist.barrier()

            epoch += 1

    # all epochs completed
    if rank == 0:
        print("END TRAINING")


def log_params(writer, model, batches_seen):
    for (
        param_name,
        param,
    ) in model.named_parameters():
        writer.add_histogram(
            f"param_vs_batch/{param_name}",
            param.data,
            batches_seen,
            bins=512,
        )


def log_grad_norm(parameters, name, writer, opt_step):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    writer.add_scalar(f"total_grad_norm/{name}", total_norm, opt_step)


def sync_grads(model: to.nn.Module, n: int):
    all_grads_list = [param.grad.view(-1) for param in model.parameters()]
    all_grads = to.cat(all_grads_list)
    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
    all_grads.div_(n)
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad.data.copy_(
            all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        offset += numel
