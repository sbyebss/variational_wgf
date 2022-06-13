import pytorch_lightning as pl
from src.callbacks.diffusion import Diffusion
from src.viz.points import compare_highd_kde_scatter
from src.callbacks.ou_callbacks import DataCb
from src.logger.jam_wandb import prefix_metrics_keys
from src.callbacks.metric_zoo.ksd import KSD
from src.callbacks.metric_zoo.target_funcional import kl_variational_func
from src.models.data_updation import new_pk_gmm_generator
import jammy.io as io
import torch
import copy
import os


class GMMprocess_Error_Cb(DataCb):
    def __init__(self, work_dir, solve_ksd=False) -> None:
        super().__init__()
        self.work_dir = work_dir
        self.solve_ksd = solve_ksd

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        sample_fit_beta = trainer.datamodule.mu_sample(10000).cpu()
        self.ksd = KSD(trainer.datamodule.q_log_prob, samples_fit_beta=sample_fit_beta)
        pl_module.log_dict({"error_gmm/beta": self.ksd.k_method.beta})
        pl_module.diff = Diffusion(
            trainer.datamodule.p0.sample_n,
            trainer.datamodule.p0.log_prob)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.epoch % pl_module.cfg.epochs_per_Pk == 0:
            pl_module.iterate_dataloader()
            compare_highd_kde_scatter(
                trainer.datamodule.pk_data,
                trainer.datamodule.q_gmm.sample_n, f"epoch{pl_module.epoch}.png")
            stein_divg = self.ksd(trainer.datamodule.pk_data[:3000].to(
                pl_module.device))

            assert pl_module.cfg.smooth_h
            vari_targ = kl_variational_func(
                trainer.datamodule.pk_data[:50000].to(pl_module.device), trainer.datamodule.last_mu_gauss, pl_module.h_net, trainer.datamodule.q_log_prob)

            io.fs.dump(f'ksd_{pl_module.idx_pk_dist}.pth', stein_divg)
            io.fs.dump(f'vari_targ_{pl_module.idx_pk_dist}.pth', vari_targ)

            log_info = prefix_metrics_keys(
                {"ksd": stein_divg, "vari_targ": vari_targ}, "error_gmm")
            pl_module.log_dict(log_info)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        beta_path = self.work_dir + f"/paper_plots/beta_values/beta_{trainer.datamodule.dims}.pth"
        if os.path.exists(beta_path):
            beta = io.load(beta_path)
            self.ksd = KSD(trainer.datamodule.q_log_prob, beta=beta)
        else:
            sample_fit_beta = trainer.datamodule.q_sample(10000).cpu()
            self.ksd = KSD(trainer.datamodule.q_log_prob, samples_fit_beta=sample_fit_beta)
            io.fs.dump(beta_path, self.ksd.k_method.beta)

        trainer.datamodule.to_device(pl_module.device)
        data_path = os.path.join(
            self.work_dir,
            f"logs/papers/gmm/multirun/{trainer.datamodule.dims}/{os.environ['PL_GLOBAL_SEED']}"
        )
        if os.path.exists(f'{data_path}/pk_data_40.pth'):
            pk_data = io.load(f'{data_path}/pk_data_40.pth')
        else:
            iterated_map = copy.deepcopy(pl_module.map_t)
            pk_data = new_pk_gmm_generator(
                iterated_map, 40, trainer.datamodule.p0, 30000, path=f'{data_path}/')
            torch.save(pk_data, f'{data_path}/pk_data_40.pth')
        torch.save(trainer.datamodule.q_gmm, f'{data_path}/target_dist.pth')
        if self.solve_ksd:
            tmp_result = 0.
            for idx_repeat in range(10):
                tmp_result += self.ksd(
                    pk_data[3000 * idx_repeat:3000 *
                            (idx_repeat + 1)].to(pl_module.device))
            stein_divg = tmp_result / 10.
            # stein_divg = self.ksd(pk_data[::10000].to(pl_module.device))
            io.fs.dump(
                f'{data_path}/ksd_40.pth', stein_divg)
