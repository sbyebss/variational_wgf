import hydra
from omegaconf import open_dict
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.models.loss_zoo_a import get_capital_a, get_capital_a_init
from src.models.loss_zoo_b import get_capital_b, w2_loss, constraint_loss
from src.logger.jam_wandb import prefix_metrics_keys
from src.models.data_updation import map_forward
from src.networks.dense_icnn import id_pretrain_model
import copy


def pretrain_model(trainer, pl_module):
    if type(pl_module.map_t).__name__ == "DenseICNN":
        if not pl_module.cfg.skip_pretrain:
            return id_pretrain_model(
                pl_module.map_t, trainer.datamodule.p0_sample, lr=pl_module.cfg['pretrain_lr'],
                n_max_iterations=4000, batch_size=pl_module.cfg.T_net.batch_size, verbose=True)
    print("map_t is not Dense ICNN, don't need to pretrain model. Or asked to skip pretrain")
    return pl_module.map_t


class BaseModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.a_func = get_capital_a(cfg.energy_type)
        self.b_func = get_capital_b(cfg.energy_type)
        if cfg.crank_nicolson:
            self.a_init = get_capital_a_init(cfg.energy_type)

        self.instantiate_map_t()
        self.instantiate_h()

        self.instantiate_spec_property()
        self.automatic_optimization = False

    def instantiate_h(self):
        self.h_net = hydra.utils.instantiate(self.cfg.h_net)

    def instantiate_map_t(self):
        self.is_icnn = False
        if self.cfg.T_net._target_ == "src.networks.dense_icnn.DenseICNN":
            with open_dict(self.cfg.T_net):
                self.cfg.T_net["hidden_layer_sizes"] = [
                    self.cfg.T_net['hidden_dim']] * self.cfg.T_net['num_layer']
                del self.cfg.T_net['hidden_dim'], self.cfg.T_net['num_layer'],
        if "ICNN" in self.cfg.T_net._target_:
            self.is_icnn = True
        self.map_t = hydra.utils.instantiate(self.cfg.T_net)
        self.map_t.is_icnn = self.is_icnn

    def instantiate_spec_property(self):
        self.t_tr_elaps, self.t_tr_strt = 0.0, 0.0

    @property
    def epoch(self):
        return self.current_epoch + 1

    @property
    def idx_pk_dist(self):
        return self.epoch // self.cfg.epochs_per_Pk

    @property
    def diffusion_time(self):
        return self.idx_pk_dist * self.cfg.step_a

    def on_pretrain_routine_start(self) -> None:
        self.map_t = pretrain_model(self.trainer, self)

    def training_step(self, y_data, batch_idx) -> None:
        z_data = self.sample_z_data(y_data.shape[0])
        optimizer_t, optimizer_h = self.optimizers()
        self.opt_lambda(y_data, z_data, optimizer_h)
        self.opt_theta(y_data, optimizer_t)

    def sample_z_data(self, num: int):
        return self.trainer.datamodule.mu_sample(num)

    def opt_lambda(self, y_tensor, z_tensor, lambda_opt):
        for _ in range(self.cfg.N_inner_ITERS):
            lambda_opt.zero_grad()
            loss, loss_info = self.loss_lambda(y_tensor, z_tensor)
            self.manual_backward(loss)
            lambda_opt.step()
        self.log_dict(loss_info)

    def opt_theta(self, y_tensor, theta_opt):
        for _ in range(self.cfg.N_outer_ITERS):
            theta_opt.zero_grad()
            loss, loss_info = self.loss_theta(y_tensor)
            self.manual_backward(loss)
            theta_opt.step()
            self.on_end_map_t_step()
        self.log_dict(loss_info)

    def on_end_map_t_step(self):
        if type(self.map_t).__name__ == "DenseICNN":
            self.map_t.convexify()

    def loss_lambda(self, y_tensor, z_tensor):
        # I wanna fix map_t, and get the gradient w.r.t. input
        if self.is_icnn:
            tx = map_forward(self.map_t, y_tensor)
        else:
            with torch.no_grad():
                tx = map_forward(self.map_t, y_tensor)
        a_loss, a_info = self.real_a_func(tx, self.h_net, opt="h_net")
        b_loss = self.real_b_func(z_tensor, self.h_net)
        lambda_loss = -a_loss + b_loss
        log_info = self.log_lambda(a_info, b_loss, lambda_loss=lambda_loss)
        return lambda_loss, log_info

    def log_lambda(self, a_info, b_loss, lambda_loss=None):
        return prefix_metrics_keys({
            **a_info,
            "b_loss": b_loss,
            "h_loss": -lambda_loss},
            "lambda")

    def nn_loss(self):
        if type(self.map_t).__name__ == "ICNN":
            return 10 * constraint_loss(self.map_t.positive_params)
        else:
            return 0

    def loss_theta(self, y_tensor):
        ty = map_forward(self.map_t, y_tensor)
        loss1 = w2_loss(y_tensor, ty, self.cfg.step_a)
        for param in self.h_net.parameters():
            param.requires_grad = False
        if self.cfg.crank_nicolson:
            a_curr_loss, a_curr_info = self.real_a_func(ty, self.h_net, opt="map_t")
            if self.idx_pk_dist > 1:
                a_last_loss, a_last_info = self.real_a_func(
                    ty, self.last_h_net, cached_gamma=True, opt="map_t")
            else:
                a_last_loss, a_last_info = self.real_a_func_init(ty)
            a_loss = a_curr_loss * self.cfg.curr_weight + a_last_loss * self.cfg.last_weight
            a_info = {**prefix_metrics_keys({**a_curr_info},
                                            "current"),
                      **prefix_metrics_keys({**a_last_info},
                                            "previous")}
        else:
            a_loss, a_info = self.real_a_func(ty, self.h_net, opt="map_t")
        for param in self.h_net.parameters():
            param.requires_grad = True
        nn_loss = self.nn_loss()
        theta_loss = loss1 + a_loss + nn_loss
        log_info = self.log_theta(a_info, loss1, a_loss, theta_loss, nn_loss)
        return theta_loss, log_info

    def log_theta(self, a_info, loss1, a_loss, theta_loss, nn_loss):
        return prefix_metrics_keys({
            **a_info,
            "w2_loss": loss1,
            "a_loss": a_loss,
            "theta_loss": theta_loss,
            "nn_loss": nn_loss
        },
            "theta")

    def configure_optimizers(self):
        optimizer_map = optim.Adam(self.map_t.parameters(
        ), lr=self.cfg.lr_g, weight_decay=self.cfg.weight_decay)
        optimizer_h = optim.Adam(self.h_net.parameters(), lr=self.cfg.lr_h,
                                 weight_decay=self.cfg.weight_decay)
        if self.cfg.schedule_learning_rate:
            return [optimizer_map, optimizer_h], [StepLR(optimizer_map, step_size=self.cfg.lr_schedule_epoch, gamma=self.cfg.lr_schedule_scale_t), StepLR(optimizer_h, step_size=self.cfg.lr_schedule_epoch, gamma=self.cfg.lr_schedule_scale_h)]
        else:
            return optimizer_map, optimizer_h

    def iterate_dataloader(self):
        if self.is_icnn:
            self.diff.Ds.append(copy.deepcopy(self.map_t))
        torch.save(self.map_t.state_dict(), f'map_{self.idx_pk_dist}.pth')
        torch.save(self.h_net.state_dict(), f'h_{self.idx_pk_dist}.pth')
        iterated_map = copy.deepcopy(self.map_t)
        pk = self.new_pk_generator(iterated_map)

        self.trainer.datamodule.update_pk(pk)
        self.trainer.datamodule.update_gamma(pk)
        torch.save(pk, f'pk_data_{self.idx_pk_dist}.pth')
        if self.cfg.crank_nicolson:
            self.last_h_net = copy.deepcopy(self.h_net)
            for param in self.last_h_net.parameters():
                param.requires_grad = False
