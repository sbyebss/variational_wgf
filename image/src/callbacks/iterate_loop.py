import functools
import os

import jammy.io as jio
import pytorch_lightning as pl  # pylint: disable=unused-import
import torch as th
import torch_fidelity
from jamtorch.data.meta import merge_tree, num_to_groups
from pytorch_lightning import Callback
from torchvision.utils import save_image
from tqdm.auto import tqdm


def batch_run_data(func, batch_size=100, is_tqdm=False):
    @functools.wraps(func)
    def new_func(data, *args, **kwargs):
        total_batch = data.shape[0]
        arr = num_to_groups(total_batch, batch_size)
        rtn = []
        cur_idx = 0
        if is_tqdm:
            arr = tqdm(arr)
        for cur_batch_size in arr:
            cur_data = data[cur_idx : cur_idx + cur_batch_size]
            rtn.append(func(cur_data, *args, **kwargs))
            cur_idx += cur_batch_size
        return merge_tree(rtn)

    return new_func


def save_seperate_imgs(sample, sample_path, cnt):
    batch_size = len(sample)
    for i in range(batch_size):
        save_image((sample[i] + 1.0) / 2, os.path.join(sample_path, f"{cnt:07d}.png"))
        cnt += 1


class IterateLoopCb(Callback):
    def __init__(self, epoch_per_iter):
        self.epoch_per_iter = epoch_per_iter
        jio.mkdir("dataset")
        jio.mkdir("network")
        jio.mkdir("fid_imgs")
        self.fid_path = "fid_imgs"

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.shuffle_last_data()
        if (pl_module.current_epoch + 1) % self.epoch_per_iter == 0:
            idx = (pl_module.current_epoch + 1) // self.epoch_per_iter
            jio.dump(f"dataset/{idx-1:03d}.pth", pl_module.last_data)

            def fn_impl(data):
                data = data.to(pl_module.device)
                return pl_module.forward(data).cpu()

            fn_local = batch_run_data(fn_impl, 256, True)
            with th.no_grad():
                new_data = fn_local(pl_module.last_data)
            pl_module.last_data = new_data
            jio.dump(f"network/{idx-1:03d}.pth", pl_module.generator.state_dict())
            pl_module.log("idx_cnt", idx - 1)
            # TODO: only support cifar10 out of box
            if trainer.datamodule.cfg.dataset == "CIFAR10":
                if idx > 10:
                    self.check_fid(pl_module)
            # pl_module.reload_modules()

    def check_fid(self, pl_module: "pl.LightningModule") -> None:
        save_seperate_imgs(pl_module.last_data[:50_000], self.fid_path, 0)
        metric = torch_fidelity.calculate_metrics(
            input1=self.fid_path,
            input2="cifar10-train",
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            verbose=False,
        )
        pl_module.log_dict(metric)
