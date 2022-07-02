import torch as th
from hydra.utils import instantiate
from importlib_metadata import collections
from omegaconf import DictConfig
from torch import nn

from .base_model import BaseModel
from .utils import bufcnt, get_weights_init_fn


# pylint: disable=too-many-ancestors, arguments-differ
class SNModel(BaseModel):
    def __init__(self, cfg: DictConfig):
        self.last_data = None
        super().__init__(cfg)
        self.automatic_optimization = False
        self.weights_init_fn = None
        self.skip_disc = False

    def instantiate(self):
        self.generator = instantiate(self.cfg.generator)
        self.discriminator = instantiate(self.cfg.discriminator)
        self.last_data_size = int(2 * self.cfg.data_length)
        self.last_data = th.rand((self.last_data_size, *self.z_shape)) * 2 - 1.0

    def on_pretrain_routine_start(self) -> None:
        super().on_pretrain_routine_start()
        # FIXME: fix the type
        self.weights_init_fn = get_weights_init_fn()
        self.generator.apply(self.weights_init_fn)
        self.discriminator.apply(self.weights_init_fn)

    def training_step(self, batch, batch_idx):
        real_imgs = self.data_transform_fn(batch)
        g_optimizer, d_optimizer = self.optimizers()
        if not self.skip_disc:
            # opt discriminator:
            d_optimizer.zero_grad()
            d_loss = self.opt_d(real_imgs, batch_idx)
            self.manual_backward(d_loss)
            th.nn.utils.clip_grad_value_(
                self.discriminator.parameters(), clip_value=1.0
            )
            d_optimizer.step()

        # opt generator
        if self.global_step % self.cfg.n_critic == 0 or self.skip_disc:
            g_optimizer.zero_grad()
            g_loss = self.opt_g(real_imgs)
            self.manual_backward(g_loss)
            th.nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=1.0)
            g_optimizer.step()

    def opt_d(self, real_imgs, batch_idx):
        real_validity = self.discriminator(real_imgs)
        with th.no_grad():
            z_vars = self.get_z_sample(real_validity.shape[0], idx=batch_idx)
            fake_imgs = self.forward(z_vars)
            del z_vars
        fake_validity = self.discriminator(fake_imgs)
        d_loss_real = th.mean(nn.ReLU(inplace=True)(1.0 - real_validity))
        d_loss_fake = th.mean(nn.ReLU(inplace=True)(1.0 + fake_validity))
        d_loss = d_loss_real + d_loss_fake
        self.log_dict(
            {
                "d_loss/real": d_loss_real,
                "d_loss/fake": d_loss_fake,
                "d_loss/loss": d_loss,
            },
            on_step=True,
            prog_bar=True,
        )
        cur_flag = d_loss.item() < 1e-4  # or g_loss.item() > 50
        self.skip_disc = bufcnt(cur_flag) > 5
        self.log("skip_disc", self.skip_disc)
        return d_loss

    def opt_g(self, real_imgs):
        batch_size = real_imgs.shape[0] * 2
        del real_imgs
        fake_inputs = self.get_z_sample(batch_size)
        fake_imgs = self.generator(fake_inputs)
        fake_validity = self.discriminator(fake_imgs)
        g_fake_loss = -th.mean(fake_validity)
        w2_loss = (
            th.mean((fake_inputs - fake_imgs).pow(2).flatten(start_dim=1).sum(dim=-1))
            * self.cfg.w2_weight
        )
        g_loss = g_fake_loss + w2_loss
        self.log_dict(
            {"g_loss/loss": g_loss, "g_loss/fake": g_fake_loss, "g_loss/w2": w2_loss},
            on_step=True,
            prog_bar=True,
        )
        cur_flag = g_fake_loss.item() > 10
        self.skip_disc = bufcnt(cur_flag) > 5
        return g_loss

    def configure_optimizers(self):
        # in cifar we betas=(0.0, 0.9)
        # beta_1 ignore moments, use current gradient use moments
        # beta_2 large accumulate of the gradient magnitude
        opt_g = th.optim.Adam(
            self.generator.parameters(),
            lr=self.cfg.g_lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        opt_d = th.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.d_lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        return opt_g, opt_d

    def forward(self, z_vars=None, num_sample=None):
        if z_vars is not None:
            return self.generator(z_vars)
        else:
            return self.sample_n(num_sample)

    def sample_n(self, num_sample):
        return self.generator(self.get_z_sample(num_sample))

    def logp(self, x):
        return self.discriminator(x)

    def reload_modules(self):
        self.generator.apply(self.weights_init_fn)
        self.discriminator.apply(self.weights_init_fn)
        for opt in self.optimizers():
            opt.state = collections.defaultdict(dict)  # reset state

    def get_z_sample(self, batch_size, idx=None):
        if idx is None:
            perm = th.randperm(self.last_data_size)[:batch_size]
            return (self.last_data[perm]).to(self.device)
        else:
            start_idx = batch_size * idx
            return (self.last_data[start_idx : start_idx + batch_size]).to(self.device)

    def shuffle_last_data(self):
        perm = th.randperm(self.last_data_size)
        self.last_data = self.last_data[perm]
