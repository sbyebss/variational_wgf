import pytorch_lightning as pl
from src.viz.points import compare_highd_kde_scatter, plot_porous_1d
from src.callbacks.ou_callbacks import DataCb
from src.callbacks.diffusion import Diffusion
from src.logger.jam_wandb import prefix_metrics_keys
from src.callbacks.metric_zoo.porous_media import *
from src.callbacks.metric_zoo.target_funcional import gen_entropy_variational_func, gen_entropy_exact_func
from jammy import io


class Plot_Compare_Porous_Cb(DataCb):
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.diff = Diffusion(
            trainer.datamodule.p0_sample,
            trainer.datamodule.density_logp0_unnml, n_max_prop=trainer.datamodule.cfg.batch_size)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.epoch % pl_module.cfg.epochs_per_Pk == 0:
            current_time = self.log_pk_info(trainer, pl_module)

            def density_pt(data):
                return trainer.datamodule.density_pt_unnml(data, current_time).detach().cpu()

            def gt_sampler(num):
                return trainer.datamodule.gt_sample(num, current_time)
            if pl_module.is_icnn:
                def density_fit(data):
                    data = data.to(pl_module.device)
                    return pl_module.diff.log_prob(data).exp().detach().cpu()
            else:
                density_fit = None

            self.plot_h(trainer, pl_module, gt_sampler, density_fit, density_pt)
            if pl_module.epoch % pl_module.cfg.epochs_per_eval == 0:
                self.log_entropy(trainer, pl_module, gt_sampler, density_pt)
                self.log_symkl(trainer, pl_module, density_fit, density_pt, gt_sampler)

    def log_entropy(self, trainer, pl_module, gt_sampler, density_pt):
        assert pl_module.cfg.smooth_h == False and pl_module.cfg.non_log_ratio
        vari_targ = gen_entropy_variational_func(
            trainer.datamodule.pk_data[:50000].to(pl_module.device), trainer.datamodule.last_q_uniform, trainer.datamodule.last_q_density, pl_module.h_net, trainer.datamodule.cfg.porous_m)
        exact_targ = gen_entropy_exact_func(
            gt_sampler, density_pt, trainer.datamodule.cfg.porous_m)

        io.fs.dump(f'vari_targ_{pl_module.idx_pk_dist}.pth', vari_targ)
        io.fs.dump(f'exact_targ_{pl_module.idx_pk_dist}.pth', exact_targ)

        log_info = prefix_metrics_keys(
            {"exact_targ": exact_targ, "vari_targ": vari_targ}, "entropy_porous")
        pl_module.log_dict(log_info)

    def log_symkl(self, trainer, pl_module, density_fit, density_pt, gt_sampler):
        if density_fit != None:
            log10symkl = log10symkl_icnn(
                density_pt, gt_sampler, trainer.datamodule.pk_data, density_fit)
            io.fs.dump(f'log10symkl_{pl_module.idx_pk_dist}.pth', log10symkl)
            log_info = prefix_metrics_keys({"log10symkl": log10symkl}, "error_porous")
            pl_module.log_dict(log_info)

    def log_pk_info(self, trainer, pl_module):
        current_time = trainer.datamodule.cfg.t0 + pl_module.idx_pk_dist * pl_module.cfg.step_a
        max_pk_cached = trainer.datamodule.pk_data.abs().max().item()
        pl_module.iterate_dataloader()
        max_pk = trainer.datamodule.pk_data.abs().max().item()
        inflation_ratio = max_pk / max_pk_cached

        log_info = prefix_metrics_keys({
            "k": pl_module.idx_pk_dist,
            "current_time": current_time,
            "pk_max": max_pk,
            "inflation_ratio": inflation_ratio,
            "pk_mean": trainer.datamodule.pk_data.mean().item(),
            "pk_std": trainer.datamodule.pk_data.std().item(),
            "density_q": trainer.datamodule.q_density
        },
            "porous_aux")
        pl_module.log_dict(log_info)
        return current_time

    def plot_h(self, trainer, pl_module, gt_sampler, density_fit, density_pt):
        if trainer.datamodule.dims > 1:
            compare_highd_kde_scatter(
                trainer.datamodule.pk_data,
                gt_sampler, f"epoch{pl_module.epoch}.png", plot_size=1, levels=5)
            h_diff_mean, h_diff_max = compare_highd_h(
                trainer.datamodule.q_sample, pl_module.h_net,
                trainer.datamodule.pk_data, trainer.datamodule.last_q_density)
            log_info = prefix_metrics_keys(
                {"h_diff_mean": h_diff_mean, "h_diff_max": h_diff_max}, "h_error")
            pl_module.log_dict(log_info)
        else:
            plot_porous_1d(
                trainer.datamodule.pk_data,
                density_fit, density_pt, trainer.datamodule.last_q_density, pl_module.h_net, pl_module.cfg.plot_bound, pl_module.epoch)

    # TODO: tasks:
    # KSD: save the target distributions and particles somewhere
    # and calculate the KSD in ipynb notebook
