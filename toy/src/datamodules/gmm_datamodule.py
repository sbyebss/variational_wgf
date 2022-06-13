from typing import Optional
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch.distributions as TD
from src.utils.th_utils import deep_to
import copy


def given_singular_spd_cov(n_dim, range_sing=[0.5, 1]):
    A = np.random.rand(n_dim, n_dim)
    U, _, V = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(
        U, np.diag(range_sing[0] + np.random.rand(n_dim) * (range_sing[1] - range_sing[0]))), V)
    return X


class BaseDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.pk_data, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.num_workers)

    def mu_sample(self, n: int):
        return self.mu_gauss.sample_n(n)

    def mu_log_prob(self, data: torch.Tensor):
        return self.mu_gauss.log_prob(data)

    def last_mu_log_prob(self, data: torch.Tensor):
        return self.last_mu_gauss.log_prob(data)

    def update_pk(self, new_pk_data):
        self.pk_data = new_pk_data

    def update_gamma(self, new_pk_data: torch.Tensor):
        assert new_pk_data.shape == torch.Size([self.train_size, self.dims])
        mean_mu = new_pk_data.mean(axis=0)
        cov_mu = torch.cov(new_pk_data.T)
        assert mean_mu.shape == torch.Size([self.dims])
        assert cov_mu.shape == torch.Size([self.dims, self.dims])

        self.last_mu_gauss = copy.deepcopy(self.mu_gauss)
        mu_gauss = TD.MultivariateNormal(mean_mu, cov_mu)
        self.mu_gauss = deep_to(mu_gauss, self.mu_gauss.loc.device)


class GMMDataModule(BaseDataModule):
    """
    A DataModule implements 5 key methods:
    - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
    - setup (things to do on every accelerator in distributed mode)
    - train_dataloader (the training dataloader)
    - val_dataloader (the validation dataloader(s))
    - test_dataloader (the test dataloader(s))
    """

    def __init__(self, cfg=None):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.train_size = self.cfg.n_train_samples
        self.dims = self.cfg.input_dim

        if cfg.num_gmm_component > 1:
            mix = TD.Categorical(torch.ones(cfg.num_gmm_component,))
            comp = TD.Independent(TD.Normal(
                torch.FloatTensor(
                    cfg.num_gmm_component,
                    self.dims).uniform_(-self.cfg.target_uf_bound, self.cfg.target_uf_bound),
                torch.ones(cfg.num_gmm_component, self.dims)), 1)
            self.q_gmm = TD.MixtureSameFamily(mix, comp)
        else:
            self.q_gmm = TD.MultivariateNormal(torch.randn(
                self.dims), torch.from_numpy(given_singular_spd_cov(self.dims)).float())

        self.mu_gauss = TD.MultivariateNormal(
            torch.zeros(self.dims), self.cfg.mu_var * torch.eye(
                self.dims))
        self.p0 = TD.MultivariateNormal(
            torch.zeros(self.dims), self.cfg.p0_std**2 * torch.eye(
                self.dims))
        self.pk_data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.cfg.mu_equal_q == 0 or self.cfg.p0_equal_q == 0
        self.pk_data = self.p0_sample(self.train_size)

    def p0_sample(self, n):
        return self.p0.sample_n(n)

    def q_sample(self, n: int):
        return self.q_gmm.sample_n(n)

    def p0_log_prob(self, data: torch.Tensor):
        return self.p0.log_prob(data)

    def q_log_prob(self, data: torch.Tensor):
        return self.q_gmm.log_prob(data)

    def to_device(self, device):
        self.mu_gauss = deep_to(self.mu_gauss, device)
        self.q_gmm = deep_to(self.q_gmm, device)
        self.p0 = deep_to(self.p0, device)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.pk_data[:100], batch_size=100, shuffle=False, num_workers=4)

    @classmethod
    def instantiate(cls, merge_str=None):
        from omegaconf import OmegaConf

        default = """
        batch_size: 100
        num_workers: 0
        pin_memory: False
        num_gmm_component: 9
        target_uf_bound: 5
        input_dim: 12
        mu_var: 3.0
        p0_std: 1.5
        n_train_samples: 102400

        mu_equal_q: False
        p0_equal_q: False
        """
        cfg = OmegaConf.create(default)
        if merge_str is not None:
            merge_cfg = OmegaConf.create(merge_str)
            cfg = OmegaConf.merge(cfg, merge_cfg)
        data_module = cls(cfg)
        return data_module


class GMMTwoSampleDataModule(GMMDataModule):
    def setup_train_data(self):
        self.pk_data = torch.randn(self.train_size, self.dims) * self.cfg.p0_std
        self.q_data = self.q_sample(self.train_size)

    def train_dataloader(self):
        loader_pk = DataLoader(
            self.pk_data, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=self.cfg.num_workers)
        loader_q = DataLoader(
            self.q_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
        )
        return [loader_pk, loader_q]
