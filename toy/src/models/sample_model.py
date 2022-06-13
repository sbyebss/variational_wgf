from .kl_gmm_model import BaseModule
from src.models.data_updation import new_pk_image_generator, new_pk_gmm_generator
import copy
from src.logger.jam_wandb import prefix_metrics_keys
from src.viz.img import save_tensor_imgs
import torch


class SampleModule(BaseModule):
    def real_a_func(self, tx, h_net, **kwargs):
        return self.a_func(tx, h_net)

    def real_b_func(self, z_tensor, h_net):
        return self.b_func(h_net, z_tensor)

    def training_step(self, batch_data, batch_idx) -> None:
        y_data, z_data = self.get_real_data(batch_data, batch_idx)
        optimizer_t, optimizer_h = self.optimizers()
        self.opt_lambda(y_data, z_data, optimizer_h)
        self.opt_theta(y_data, optimizer_t)

    def get_real_data(self, batch_data, batch_idx):
        if self.cfg.type_q == 'image':
            d_fn = self.trainer.datamodule.data_transform
            y_data, z_data_label = batch_data
            z_data = d_fn(z_data_label[0])
            if batch_idx == 1:
                save_tensor_imgs(self.trainer.datamodule.inverse_data_transform(
                    y_data), 10, self.global_step, "batch_my")
                save_tensor_imgs(self.trainer.datamodule.inverse_data_transform(
                    z_data), 10, self.global_step, "batch_z")
            return y_data, z_data
        else:
            return batch_data

    def log_lambda(self, a_info, b_loss, lambda_loss=None):
        return prefix_metrics_keys({
            **a_info,
            "b_loss": b_loss,
            "lambda_loss(-a+b)": lambda_loss,
        },
            "lambda")

    def log_theta(self, a_info, loss1, a_loss, theta_loss, nn_loss):
        del a_info, nn_loss
        return prefix_metrics_keys({
            "w2_loss": loss1,
            "a_loss": a_loss,
            "theta_loss": theta_loss
        },
            "theta")

    def iterate_dataloader(self):
        iterated_map = copy.deepcopy(self.map_t)
        torch.save(self.map_t.state_dict(), f'map_{self.idx_pk_dist}.pth')
        if self.cfg.type_q == 'image':
            pk = new_pk_image_generator(
                iterated_map, self.idx_pk_dist, self.trainer.datamodule.train_size, self.trainer.datamodule.dims, self.device)
        else:
            pk = new_pk_gmm_generator(
                iterated_map, self.idx_pk_dist,
                self.trainer.datamodule.p0, self.trainer.datamodule.train_size)
        self.trainer.datamodule.update_pk(pk)
