import pytorch_lightning as pl
from pytorch_lightning import Callback
import time
import jammy.io as io
from src.logger.jam_wandb import prefix_metrics_keys
from src.callbacks.metric_zoo.ou_symkl import log10symkl_gauss_mcmc, log10symkl_gauss_with_mean_cov
from src.callbacks.metric_zoo.target_funcional import ou_exact_func, ou_variation_func


class DataCb(Callback):
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(trainer.datamodule, "to_device"):
            trainer.datamodule.to_device(pl_module.device)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.epoch % pl_module.cfg.epochs_per_Pk == 0:
            pl_module.iterate_dataloader()


class OUprocess_Error_Cb(DataCb):
    def __init__(self, eval_mcmc) -> None:
        super().__init__()
        self.eval_mcmc = eval_mcmc

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.t_tr_unit_strt = time.perf_counter()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.epoch % pl_module.cfg.epochs_per_Pk == 0:
            pl_module.iterate_dataloader()
            pl_module.t_tr_unit_elaps = time.perf_counter() - pl_module.t_tr_unit_strt
            io.fs.dump(f'train_time_{pl_module.idx_pk_dist}.pth', pl_module.t_tr_unit_elaps * 2)
            self.log_entropy(trainer, pl_module)
            self.log_symkl(trainer, pl_module)

    def log_entropy(self, trainer, pl_module):
        assert pl_module.cfg.smooth_h == False and pl_module.cfg.non_log_ratio
        vari_targ = ou_variation_func(trainer.datamodule.mu_gauss, trainer.datamodule.q_gmm)
        exact_targ = ou_exact_func(pl_module.diffusion_time, trainer.datamodule.q_gmm)

        io.fs.dump(f'vari_targ_{pl_module.idx_pk_dist}.pth', vari_targ)
        io.fs.dump(f'exact_targ_{pl_module.idx_pk_dist}.pth', exact_targ)

        log_info = prefix_metrics_keys(
            {"exact_targ": exact_targ, "vari_targ": vari_targ}, "entropy_ou")
        pl_module.log_dict(log_info)

    def log_symkl(self, trainer, pl_module):
        if self.eval_mcmc:
            log10symkl = log10symkl_gauss_mcmc(
                trainer.datamodule.q_gmm,
                trainer.datamodule.mu_gauss, trainer.datamodule.pk_data, pl_module.diffusion_time)
        else:
            log10symkl = log10symkl_gauss_with_mean_cov(
                trainer.datamodule.q_gmm,
                trainer.datamodule.mu_gauss, pl_module.diffusion_time)
        io.fs.dump(f'log10symkl_{pl_module.idx_pk_dist}.pth', log10symkl)

        log_info = prefix_metrics_keys({
            "current_k": pl_module.idx_pk_dist,
            "time_per_JKOstep": pl_module.t_tr_unit_elaps,
            "log10symkl": log10symkl
        },
            "ou_process")
        pl_module.log_dict(log_info)


class Total_Time_Cb(Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.t_tr_strt = time.perf_counter()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.t_tr_elaps = time.perf_counter() - pl_module.t_tr_strt
        io.fs.dump(f'train_time.pth', pl_module.t_tr_elaps)
