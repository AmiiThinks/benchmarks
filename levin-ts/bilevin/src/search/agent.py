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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch as to
import torch.distributed as dist

from domains.domain import Domain
import models.loss_functions as loss_fns
from models.models import AgentModel


class Agent(ABC):
    def __init__(self, rank, run_name, args, model_args):
        if not self.trainable:
            return

        model_args["bidirectional"] = self.bidirectional

        model_save_path = Path(__file__).parents[2] / f"runs/{run_name}"
        model_save_path.mkdir(parents=True, exist_ok=True)
        self.save_path: Path = model_save_path

        self._model: to.jit.RecursiveScriptModule

        if args.model_path is None:
            # just use the random initialization from rank 0
            model = AgentModel(model_args)
            if args.world_size > 1:
                for param in model.parameters():
                    dist.broadcast(param.data, 0)
            self._model = to.jit.script(model)
        elif args.model_path.is_dir():
            load_model_path = args.model_path / f"model_{args.model_suffix}.pt"
            self._model = to.jit.load(load_model_path)

            if rank == 0:
                print(f"Loaded model\n  {str(load_model_path)}")
        else:
            raise ValueError("model-path argument must be a directory if given")

        if rank == 0:
            init_model = model_save_path / f"model_init.pt"
            print(f"Saving init model\n  to {str(init_model)}")
            to.jit.save(self._model, init_model)

        if args.mode == "train":
            assert self._model
            self.loss_fn = getattr(loss_fns, args.loss_fn)
            optimizer_params = [
                {
                    "params": self.model.feature_net.parameters(),
                    "lr": args.feature_net_lr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": self.model.forward_policy.parameters(),
                    "lr": args.forward_policy_lr,
                    "weight_decay": args.weight_decay,
                },
            ]
            if self.bidirectional:
                optimizer_params.append(
                    {
                        "params": self.model.backward_policy.parameters(),
                        "lr": args.backward_policy_lr,
                        "weight_decay": args.weight_decay,
                    }
                )
            self.optimizer = to.optim.Adam(optimizer_params)

    @property
    def model(self) -> to.jit.ScriptModule:
        return self._model

    def save_model(
        self,
        suffix="",
        log=True,
    ):
        path = self.save_path
        if suffix:
            path = path / f"model_{suffix}.pt"
        else:
            path = path / "model.pt"

        if log:
            print(f"Saving model\n  to {str(path)}")

        to.jit.save(self.model, path)

    @property
    @classmethod
    @abstractmethod
    def bidirectional(cls) -> bool:
        pass

    @property
    @classmethod
    @abstractmethod
    def trainable(cls) -> bool:
        pass

    @abstractmethod
    def search(self):
        pass
