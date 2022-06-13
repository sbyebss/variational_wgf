import numpy as np
from typing import Optional
import torch
from src.datamodules.gmm_datamodule import BaseDataModule
from torch.utils.data import Dataset, DataLoader
import torch.distributions as TD
from src.utils.th_utils import deep_to
import copy


def rejection_sampling(density_func, prop_sampler, prop_density, num, dim):
    target_tensor = torch.zeros([num, dim])
    remain_num = num
    while remain_num > 0:
        u_tensor = torch.rand(2 * num)
        u_tensor = (u_tensor[u_tensor > 0.])[:num]
        prop_tensor = prop_sampler(num)
        m_bound = density_func(
            0.) / prop_density(torch.Tensor([0.] * dim))
        accp_tensor = prop_tensor[u_tensor <= (
            density_func(prop_tensor) / m_bound / prop_density(prop_tensor))]
        accp_num = accp_tensor.shape[0]
        if accp_num < remain_num:
            target_tensor[-remain_num:(-remain_num + accp_num)] = accp_tensor
        else:
            target_tensor[-remain_num:] = accp_tensor[:remain_num]
        remain_num -= accp_num
    return target_tensor

# import scipy
# def rejection_sampling_old(density_func, prop_sampler, prop_density, num, dim, bound=0.3):
#     '''
#     only 1d
#     '''
#     x_tensor = torch.linspace(-bound, bound, 2000).reshape(-1, 1)
#     sqrt_p = torch.sqrt(density_func(x_tensor))
#     x_sqrt_p = x_tensor.reshape(-1) * sqrt_p
#     umax = max(sqrt_p)
#     vmin, vmax = min(x_sqrt_p), max(x_sqrt_p)
#     umax, vmin, vmax = umax.item(), vmin.item(), vmax.item()

#     return torch.from_numpy(scipy.stats.rvs_ratio_uniforms(
#         density_func, umax, vmin, vmax, size=num)).reshape(-1, 1).float()


class BarenblattDataModule(BaseDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.train_size = self.cfg.n_train_samples
        self.dims = self.cfg.input_dim

        self.q_uniform = TD.Independent(TD.Uniform(
            -cfg.q_bound_scale * cfg.p0_bound * torch.ones(self.dims), cfg.q_bound_scale * cfg.p0_bound * torch.ones(
                self.dims)), 1)
        self.q_density = self.q_uniform.log_prob(torch.zeros(self.dims)).exp().item()
        self.pk_data: Optional[Dataset] = None

        self.alpha = self.dims / (self.dims * (cfg.porous_m - 1) + 2)
        self.beta = self.alpha / self.dims
        self.k_value = self.alpha * (cfg.porous_m - 1) / (2 * cfg.porous_m * self.dims)
        self.c_const = self.k_value * cfg.p0_bound**2 * self.cfg.t0**(-2 * self.beta)
        # print("c_constant=", self.c_const)
        if self.dims == 1:
            self.prop = TD.MultivariateNormal(torch.zeros(
                self.dims), cfg.p0_bound * 4 * torch.eye(self.dims))
        elif self.dims < 4:
            self.prop = TD.MultivariateNormal(torch.zeros(
                self.dims), cfg.p0_bound * 2 * torch.eye(self.dims))
        else:
            self.prop = TD.MultivariateNormal(torch.zeros(
                self.dims), cfg.p0_bound * torch.eye(self.dims))

    def density_logp0_unnml(self, x):
        return (self.density_p0_unnml(x)).log()

    def density_p0_unnml(self, x):
        return self.density_pt_unnml(x, t=self.cfg.t0)

    def density_pt_unnml(self, x, t):
        if type(x) is torch.Tensor:
            x_norm = torch.norm(x, dim=1)**2
        elif type(x) is float or np.ndarray:
            x_norm = x**2
        else:
            raise Exception
        inside_relu = self.c_const - self.k_value * \
            x_norm * t**(-2 * self.beta)
        return t**(-self.alpha) * (inside_relu * (inside_relu > 0))**(1 / (self.cfg.porous_m - 1))

    def setup(self, stage: Optional[str] = None) -> None:
        self.pk_data = self.p0_sample(self.train_size)

    def train_dataloader(self):
        return DataLoader(
            self.pk_data, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.num_workers)

    def prop_density(self, data):
        return self.prop.log_prob(data).exp()

    def p0_sample(self, num):
        p0_samples = rejection_sampling(
            self.density_p0_unnml, self.prop.sample_n, self.prop_density, num, self.dims)
        return p0_samples.to(self.device) if hasattr(self, "device") else p0_samples

    def gt_sample(self, num, time):
        def density_pt(data):
            return self.density_pt_unnml(data, time)
        return rejection_sampling(density_pt, self.prop.sample_n, self.prop_density, num, self.dims)

    def q_sample(self, n):
        return self.q_uniform.sample_n(n)

    def update_pk(self, new_pk_data):
        self.pk_data = new_pk_data

    def update_gamma(self, new_pk_data: torch.Tensor):
        max_bound = new_pk_data.abs().max().item() * self.cfg.q_bound_scale
        q_uniform = TD.Independent(TD.Uniform(
            -max_bound * torch.ones(self.dims), max_bound * torch.ones(
                self.dims)), 1)
        self.last_q_density = self.q_density
        self.last_q_uniform = copy.deepcopy(self.q_uniform)

        self.q_density = q_uniform.log_prob(torch.zeros(self.dims)).exp().item()
        self.q_uniform = deep_to(q_uniform, self.q_uniform.mean.device)

    def to_device(self, device):
        self.q_uniform = deep_to(self.q_uniform, device)
        self.device = device

    @classmethod
    def instantiate(cls, merge_str=None):
        from omegaconf import OmegaConf

        default = """
        batch_size: 1024
        num_workers: 0
        pin_memory: False
        input_dim: 1
        n_train_samples: 100
        t0: 1.0e-3
        porous_m: 2        
        q_bound_scale: 1.4
        p0_bound: 0.5
        """
        cfg = OmegaConf.create(default)
        if merge_str is not None:
            merge_cfg = OmegaConf.create(merge_str)
            cfg = OmegaConf.merge(cfg, merge_cfg)
        data_module = cls(cfg)
        return data_module
