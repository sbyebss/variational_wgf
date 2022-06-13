from src.models.data_updation import new_pk_gmm_generator
from src.models.base_model import BaseModule


class GMMModule(BaseModule):
    def instantiate_spec_property(self):
        self.t_tr_elaps, self.t_tr_strt = 0.0, 0.0
        self.t_tr_unit_elaps, self.t_tr_unit_strt = 0.0, 0.0

    def real_a_func_init(self, tx):
        log_p0 = self.trainer.datamodule.p0_log_prob
        log_q = self.trainer.datamodule.q_log_prob
        return self.a_init(tx, log_p0_func=log_p0, log_q_func=log_q)

    def real_a_func(self, tx, h_net, cached_gamma=False, opt="map_t"):
        if cached_gamma:
            log_gamma = self.trainer.datamodule.last_mu_log_prob
        else:
            log_gamma = self.trainer.datamodule.mu_log_prob
        log_q = self.trainer.datamodule.q_log_prob
        return self.a_func(tx, h_net, log_gamma_func=log_gamma, log_q_func=log_q, log_ratio=not self.cfg.non_log_ratio, smooth=self.cfg.smooth_h, opt=opt)

    def real_b_func(self, z_tensor, h_net):
        return self.b_func(
            h_net, z_tensor, log_ratio=not self.cfg.non_log_ratio,
            dk_formula=self.cfg.dk_formula, smooth=self.cfg.smooth_h)

    def new_pk_generator(self, iterated_map):
        return new_pk_gmm_generator(
            iterated_map, self.idx_pk_dist,
            self.trainer.datamodule.p0, self.trainer.datamodule.train_size)

    def test_step(self, *args, **kwargs):
        pass
