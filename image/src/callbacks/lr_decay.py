from collections.abc import Iterable
from typing import Any, Optional

import pytorch_lightning as pl  # pylint: disable=unused-import
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

# pylint: disable=invalid-name,too-few-public-methods


class LinearLrDecay:
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


class LrDecayCb(Callback):
    def __init__(self) -> None:
        self.decays = None

    def on_pretrain_routine_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        optimizers = pl_module.optimizers()
        self.decays = []
        if not isinstance(optimizers, Iterable):
            optimizers = [optimizers]
        for opt in optimizers:
            first_group = next(iter(opt.param_groups))
            lr = first_group["lr"]
            cur_decay = LinearLrDecay(
                opt, lr, 0.0, pl_module.global_step, trainer.max_steps
            )
            self.decays.append(cur_decay)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        del outputs, batch_idx, unused, batch, trainer
        if pl_module.global_step % pl_module.cfg.n_critic:
            info = {}
            for i_th, cur_decay in enumerate(self.decays):
                info[f"lr/{i_th}-th"] = cur_decay.step(pl_module.global_step)
            pl_module.log_dict(info)
