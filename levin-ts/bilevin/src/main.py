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
import os
from pathlib import Path
import random
import socket
import time
from typing import Optional

import numpy as np
from test import test
import torch as to
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
import wandb

import domains
from loaders import CurriculumLoader, ProblemsBatchLoader
from search import BiBS, BiLevin, Levin
from train import train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--problemset-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with problem instances",
    )
    parser.add_argument(
        "-v",
        "--validset-path",
        type=lambda p: Path(p).absolute(),
        help="path of file with problem instances",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=lambda p: Path(p).absolute(),
        default=None,
        help="path of directory to load previously saved model(s) from",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="best",
        help="suffix of model to load, i.e. model_[suffix].pt",
    )
    parser.add_argument(
        "-l",
        "--loss-fn",
        type=str,
        default="levin_loss",
        choices=[
            "levin_loss",
            "ub_loss",
            "cross_entropy_loss",
        ],
        help="loss function",
    )
    parser.add_argument(
        "--forward-hidden-layers",
        action="store",
        nargs="+",
        default=[128],
        type=int,
        help="hidden layer sizes of forward policy",
    )
    parser.add_argument(
        "--backward-hidden-layers",
        action="store",
        nargs="+",
        default=[256, 192, 128],
        type=int,
        help="hidden layer sizes of backward policy",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="l2 regularization weight",
    )
    parser.add_argument(
        "--feature-net-lr",
        type=float,
        default=0.001,
        help="feature net learning rate",
    )
    parser.add_argument(
        "--forward-policy-lr",
        type=float,
        default=0.001,
        help="forward policy learning rate",
    )
    parser.add_argument(
        "--backward-policy-lr",
        type=float,
        default=0.001,
        help="backward policu learning rate",
    )
    parser.add_argument(
        "-g",
        "--grad-steps",
        type=int,
        default=10,
        help="number of gradient steps to be performed in each opt pass",
    )
    parser.add_argument(
        "--bootstrap-epochs",
        type=int,
        default=1,
        help="number of bootstrap epochs to train for",
    )
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=1,
        help="number of curriculum epochs to train for",
    )
    parser.add_argument(
        "--include-prev-difficulty",
        action="store_true",
        default=False,
        help="do not include previous difficulties in curriculum",
    )
    parser.add_argument(
        "--permutation-focus",
        action="store_true",
        default=False,
        help="just use the permutation problems once the bootstrap/curriculum is done",
    )
    parser.add_argument(
        "--permutation-epochs",
        type=int,
        default=1,
        help="number of permutation epochs to train for",
    )
    parser.add_argument(
        "--epoch-reduce-lr",
        type=int,
        default=9999999,
        help="reduce learning rate by a factor of 10 after this many epochs",
    )
    parser.add_argument(
        "--epoch-reduce-grad-steps",
        type=int,
        default=9999999,
        help="reduce number of grad steps by a factor of 2 after this many epochs",
    )
    parser.add_argument(
        "--epoch-begin-validate",
        type=int,
        default=1,
        help="reduce learning rate by a factor of 10 after this many epochs",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=["Levin", "BiLevin", "BiBS"],
        help="name of the search agent",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="number of processes to spawn",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=socket.gethostname(),
        help="address for multiprocessing communication",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="34567",
        help="port for multiprocessing communication",
    )
    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=32,
        help="number of problems to batch during",
    )
    parser.add_argument(
        "--expansion-budget",
        type=int,
        default=2**10,
        help="initial node expansion budget to solve a problem",
    )
    parser.add_argument(
        "--increase-budget",
        action="store_true",
        default=False,
        help="during testing (not validation), double the budget for each unsolved problem",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=300,
        help="time budget to solve a problem",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="train or test the model from model-folder using instances from problems-folder",
    )
    parser.add_argument(
        "--exp-name", type=str, default="", help="the name of this experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed of the experiment",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["disabled", "online", "offline"],
        help="track with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bilevin",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="ken-levi",
        help="the entity (team) of the wandb project",
    )
    args = parser.parse_args()
    return args


def run(rank, run_name, model_args, args, local_loader, local_valid_loader):
    is_distributed = args.world_size > 1

    if is_distributed:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
        dist.init_process_group(backend="gloo", rank=rank, world_size=args.world_size)

    if args.mode == "test":
        run_name = f"test_{run_name}"

    if rank == 0:
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="src/"),
        )
        if args.wandb_mode != "disabled":
            print(
                f"Logging with Weights and Biases\n  to {args.wandb_entity}/{args.wandb_project}/{run_name}"
            )

        print(f"Logging with tensorboard\n  to runs/docker/{run_name}\n")

        writer = SummaryWriter(f"runs/docker/{run_name}")
        arg_string = "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        )
        for arg in arg_string.splitlines()[2:]:
            arg = arg.replace("|", "", 1)
            arg = arg.replace("|", ": ", 1)
            arg = arg.replace("|", "", 1)
            print(arg)
        print()

        writer.add_text(
            "hyperparameters",
            arg_string,
        )

    else:
        writer = None

    local_seed = args.seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    to.manual_seed(local_seed)

    if args.agent == "Levin":
        agent = Levin(rank, run_name, args, model_args)
    elif args.agent == "BiLevin":
        agent = BiLevin(rank, run_name, args, model_args)
    elif args.agent == "BiBS":
        agent = BiBS(rank, run_name, args, model_args)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    if args.mode == "train":
        train(
            rank,
            agent,
            local_loader,
            writer,
            args.world_size,
            expansion_budget=args.expansion_budget,
            time_budget=args.time_budget,
            seed=local_seed,
            grad_steps=args.grad_steps,
            epoch_reduce_lr=args.epoch_reduce_lr,
            epoch_reduce_grad_steps=args.epoch_reduce_grad_steps,
            epoch_begin_validate=args.epoch_begin_validate,
            valid_loader=local_valid_loader,
        )

    elif args.mode == "test":
        test(
            rank,
            agent,
            local_loader,
            writer,
            args.world_size,
            expansion_budget=args.expansion_budget,
            increase_budget=args.increase_budget,
            print_results=True,
            validate=False,
            epoch=None,
        )

    if rank == 0:
        total_time = time.time() - start_time
        time_string = f"Total time: {total_time:.2f} seconds"
        print(time_string)
        Path(f"runs/docker/world_size-{args.world_size}-time.txt").write_text(time_string)
        writer.add_text("total_time", f"{total_time:.2f} seconds")
        writer.close()
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    is_distributed = args.world_size > 1

    if args.mode == "train":
        if args.batch_size_train < args.world_size:
            raise ValueError(
                f"batch-size-train'{args.batch_size_train}' must be >= world_size {args.world_size}"
            )
        if args.batch_size_train % args.world_size != 0:
            raise ValueError(
                f"batch-size-train '{args.batch_size_train}' must be a multiple of world_size {args.world_size}"
            )

    start_time = time.time()

    problemset_dict = json.load(args.problemset_path.open("r"))
    domain_module = getattr(domains, problemset_dict["domain_module"])
    problemset, model_args = getattr(domain_module, "parse_problemset")(problemset_dict)

    if args.validset_path:
        validset_dict = json.load(args.validset_path.open("r"))
        validset, _ = getattr(domain_module, "parse_problemset")(validset_dict)

    print(time.ctime(start_time))

    def get_loaders(problemset):
        def split(problems):
            num_problems_parsed = len(problems)
            # if num_problems_parsed < args.world_size:
            #     raise Exception(
            #         f"Number of problems '{num_problems_parsed}' must be greater than world size '{args.world_size}'"
            #     )

            problemsets = [[] for _ in range(args.world_size)]
            proc = 0
            for problem in problems:
                # this should be a redundant check, but just in case
                if problem.domain.is_goal(problem.domain.initial_state):
                    raise Exception(
                        f"Problem '{problem.id}' initial state is a goal state"
                    )

                problemsets[proc].append(problem)
                proc = (proc + 1) % args.world_size

            print(f"Parsed {num_problems_parsed} problems")

            large_size = len(problemsets[0])
            small_size = len(problemsets[-1])
            if large_size == small_size:
                print(
                    f"  Loading {large_size} into each of {args.world_size} processes"
                )
            else:
                small_ranks = 0
                while len(problemsets[small_ranks]) == large_size:
                    small_ranks += 1
                    continue

                print(
                    f"  Loading {large_size} into ranks 0-{small_ranks - 1},\n"
                    f"          {small_size} into ranks {small_ranks}-{args.world_size - 1}\n"
                )

            return problemsets

        local_batch_size = args.batch_size_train // args.world_size

        def set_id_idxs(start_idx, problems):
            for i, p in enumerate(
                problems,
                start=start_idx,
            ):
                p.id_idx = i

        if "is_curriculum" in problemset:
            # for now, all training problemsets should be curricula
            bootstrap_problemsets = split(problemset["bootstrap_problems"])
            world_bootstrap_ids = [p.id for p in problemset["bootstrap_problems"]]
            set_id_idxs(0, problemset["bootstrap_problems"])

            curriculum_problems = problemset["curriculum_problems"]
            world_curr_ids = [p.id for p in problemset["curriculum_problems"]]

            if args.include_prev_difficulty:
                set_id_idxs(len(world_bootstrap_ids), curriculum_problems)

            ppd = problemset["problems_per_difficulty"]
            if ppd % args.world_size != 0:
                raise ValueError(
                    "problems_per_difficulty must be a multiple of world_size"
                )
            num_difficulty_levels = len(problemset["curriculum"])

            curriculum_diff_ranks_split = [[] for _ in range(num_difficulty_levels)]
            for i in range(num_difficulty_levels):
                curriculum_difficulty_problems = curriculum_problems[
                    i * ppd : (i + 1) * ppd
                ]
                if not args.include_prev_difficulty:
                    set_id_idxs(0, curriculum_difficulty_problems)
                curriculum_diff_ranks_split[i] = split(curriculum_difficulty_problems)

            curriculum_problemsets = [[] for _ in range(args.world_size)]
            for i in range(args.world_size):
                for j in range(num_difficulty_levels):
                    curriculum_problemsets[i].append(curriculum_diff_ranks_split[j][i])

            permutation_problemsets = split(problemset["permutation_problems"])
            world_permutation_ids = [p.id for p in problemset["permutation_problems"]]
            start_idx = (
                len(world_bootstrap_ids) + len(world_curr_ids)
                if (args.include_prev_difficulty and not args.permutation_focus)
                else 0
            )
            set_id_idxs(
                start_idx,
                problemset["permutation_problems"],
            )

            world_curr_ids = [
                world_curr_ids[i : i + ppd] for i in range(0, len(world_curr_ids), ppd)
            ]

            loaders = []
            for rank in range(args.world_size):
                loaders.append(
                    CurriculumLoader(
                        local_bootstrap_problems=bootstrap_problemsets[rank],
                        world_bootstrap_ids=world_bootstrap_ids,
                        bootstrap_epochs=args.bootstrap_epochs,
                        curriculum=problemset["curriculum"],
                        world_problems_per_difficulty=ppd,
                        local_curriculum_problems=curriculum_problemsets[rank],
                        world_curriculum_ids=world_curr_ids,
                        curriculum_epochs=args.curriculum_epochs,
                        local_permutation_problems=permutation_problemsets[rank],
                        world_permutation_ids=world_permutation_ids,
                        permutation_epochs=args.permutation_epochs,
                        local_batch_size=local_batch_size,
                        world_size=args.world_size,
                        seed=args.seed + rank,
                        include_prev_difficulty=args.include_prev_difficulty,
                        permutation_focus=args.permutation_focus,
                    )
                )

        else:
            # this is only for loading test/valid problemsets, which always use a batch_size of 1 to
            # populate lists/tuples inside the test script
            loaders = []
            problemsets = split(problemset["problems"])
            all_ids = [p.id for p in problemset["problems"]]
            set_id_idxs(0, problemset["problems"])

            for rank in range(args.world_size):
                loaders.append(
                    ProblemsBatchLoader(
                        problems=problemsets[rank],
                        all_ids=all_ids,
                        local_batch_size=1,
                        world_size=args.world_size,
                        seed=args.seed,
                    )
                )

        return loaders

    problem_loaders = get_loaders(problemset)

    valid_loaders = None
    if args.validset_path:
        valid_loaders = get_loaders(validset)

    exp_name = f"_{args.exp_name}" if args.exp_name else ""
    problemset_params = (
        f"{args.problemset_path.parent.stem}-{args.problemset_path.stem}"
    )
    run_name = f"{problemset_dict['domain_name']}-{problemset_params}_{args.agent}-e{args.expansion_budget}-t{args.time_budget}{exp_name}_{args.seed}_{int(start_time)}"
    del problemset_dict

    model_args.update(
        {
            "kernel_size": (2, 2),
            "num_filters": 32,
            "forward_hidden_layers": args.forward_hidden_layers,
            "backward_hidden_layers": args.backward_hidden_layers,
        }
    )

    if is_distributed:
        processes = []
        for rank in range(args.world_size):
            proc = mp.Process(
                target=run,
                args=(
                    rank,
                    run_name,
                    model_args,
                    args,
                    problem_loaders[rank],
                    valid_loaders[rank] if valid_loaders else None,
                ),
            )
            problem_loaders[rank] = None
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
    else:
        assert len(problem_loaders) == 1
        run(
            0,
            run_name,
            model_args,
            args,
            problem_loaders[0],
            valid_loaders[0] if valid_loaders else None,
        )
