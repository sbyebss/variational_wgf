from .bayesian_model import BayesianModule


class GeneralEnergyModule(BayesianModule):
    def real_a_func_init(self, tx):
        p0_func = self.trainer.datamodule.density_p0_unnml
        return self.a_init(tx, p0_func=p0_func, m=self.cfg.porous_m)

    def real_a_func(self, tx, h_net, cached_gamma=False, opt="map_t"):
        if cached_gamma:
            q_func = self.trainer.datamodule.last_q_density
        else:
            q_func = self.trainer.datamodule.q_density
        return self.a_func(tx, h_net, q_func=q_func, log_ratio=not self.cfg.non_log_ratio, smooth_h=self.cfg.smooth_h, opt=opt, m=self.cfg.porous_m)

    def real_b_func(self, z_tensor, h_net):
        return self.b_func(
            h_net, z_tensor, q_func=self.trainer.datamodule.q_density,
            log_ratio=not self.cfg.non_log_ratio,
            smooth_h=self.cfg.smooth_h, m=self.cfg.porous_m)

    def sample_z_data(self, num: int):
        return self.trainer.datamodule.q_sample(num)
