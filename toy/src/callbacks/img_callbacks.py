import jammy.io as jio
import numpy as np
import pytorch_lightning as pl  # pylint: disable=unused-import
import torch as th
import torch_fidelity
# from jamtorch.data import num_to_groups
from pytorch_lightning import Callback
from pytorch_lightning.utilities.distributed import distributed_available, rank_zero_only

# from src.models.sampling import sample_sde_lm
from src.viz.img import save_tensor_imgs
# save_seperate_imgs
# pylint: disable=arguments-differ,too-many-instance-attributes


class ImgViz(Callback):
    def on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with th.no_grad():
            source = trainer.datamodule.pk_data[:100].to(pl_module.device)
            generate = pl_module.map_t(source).detach().cpu()
            save_tensor_imgs(trainer.datamodule.inverse_data_transform(
                generate), 10, pl_module.epoch, "forward_pk")

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.epoch % pl_module.cfg.epochs_per_Pk == 0:
            pl_module.iterate_dataloader()
            save_tensor_imgs(trainer.datamodule.inverse_data_transform(
                trainer.datamodule.pk_data[:100]), 10, pl_module.epoch, "sample_pk")


class FIDSample(Callback):
    def __init__(self, dir_path, num_sample, batch_size, every_n, dataset_name):
        self.path = dir_path
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.start_idx, self.end_idx = 0, num_sample
        self.groups = None
        self.every_n = every_n
        fidelity_input2 = {
            "CIFAR10": "cifar10-train",
        }
        self.fidelity_input2 = fidelity_input2[dataset_name]

        jio.mkdir("img_fid")
        jio.mkdir(self.path)

    def on_pretrain_routine_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        num_gpu = trainer.num_gpus
        rank = pl_module.global_rank
        idx = np.linspace(0, self.num_sample, num_gpu + 1, dtype=int)
        self.start_idx, self.end_idx = idx[rank], idx[rank + 1]
        self.groups = num_to_groups(self.end_idx - self.start_idx, self.batch_size)

    def on_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_step % self.every_n == self.every_n - 1:
            cnt = self.start_idx
            for cur_batch_size in self.groups:
                xs, _ = sample_sde_lm(pl_module, cur_batch_size, is_z2x=True)
                xs = trainer.datamodule.inverse_data_transform(xs)
                save_seperate_imgs(xs, self.path, cnt)
                cnt += cur_batch_size
            if distributed_available():
                th.distributed.barrier()
            self.report_fidelity(pl_module)

    @rank_zero_only
    def report_fidelity(self, pl_module: "pl.LightningModule"):
        metric = torch_fidelity.calculate_metrics(
            input1=self.path,
            input2=self.fidelity_input2,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
        )
        log_dict = {f"img_fid/{k}": v for k, v in metric.items()}
        for key, value in log_dict.items():
            pl_module.log(key, value, rank_zero_only=True, sync_dist=False)
        jio.dump(f"img_fid/img_fieldty_{pl_module.global_step:07d}.pkl", metric)
