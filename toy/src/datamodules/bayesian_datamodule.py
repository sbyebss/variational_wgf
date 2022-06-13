from typing import Optional
import numpy as np
import torch
from src.datamodules.gmm_datamodule import BaseDataModule
import torch.distributions as TD
from torch.utils.data import DataLoader, Dataset
from src.utils.th_utils import deep_to
from src.datamodules.datasets.bayesian_dataset import BinaryDataset, GunnarRaetschBenchmarks


def get_train_test_datasets(name, split=True, torch_split_rseed=42):
    _available_ds = [
        'covtype',
        'german',
        'diabetis',
        'twonorm',
        'ringnorm',
        'banana',
        'splice',
        'waveform',
        'image']
    assert name in _available_ds
    if name in ['covtype', 'diabetis', 'twonorm', 'ringnorm']:
        dataset = BinaryDataset(name, unverified_ssl_enable=True)
    if name in ['image', 'german', 'banana', 'splice', 'waveform']:
        dataset = GunnarRaetschBenchmarks().get_dataset(name)
    if not split:
        return dataset
    if torch_split_rseed is not None:
        torch.random.manual_seed(torch_split_rseed)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    # divide into train and test subsets
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_len, test_len])
    return dataset, train_ds, test_ds


class BayesianDataModule(BaseDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.train_size = self.cfg.n_train_samples

        dataset, train_ds, self.test_ds = get_train_test_datasets(cfg['ds_name'])
        self.train_dl = DataLoader(train_ds, batch_size=cfg['data_batch_size'], shuffle=True)
        self._dataloader_iter = iter(self.train_dl)
        self.n_features = dataset.n_features
        self.dims = dataset.n_features + 1

        self.n_data_samples_drawn = 0
        self.n_data_epochs_drawn = 0

        self.mu_gauss = TD.MultivariateNormal(
            torch.zeros(self.dims), self.cfg.mu_var * torch.eye(
                self.dims))
        self.gamma0 = TD.Gamma(torch.tensor(1.), torch.tensor(100.))
        self.normal0 = TD.Normal(torch.tensor(0.), torch.tensor(1.))
        self.pk_data: Optional[Dataset] = None

    @property
    def len_dataset(self):
        return len(self.train_dl.dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        del stage
        self.pk_data = self.p0_sample(self.train_size)

    def p0_sample(self, num):
        alpha_sample = self.gamma0.sample((num,)).view(-1, 1)
        if self.cfg.clip_alpha is not None:
            alpha_sample = torch.clamp(
                alpha_sample, np.exp(-self.cfg.clip_alpha), np.exp(self.cfg.clip_alpha))
        omega_sample = self.normal0.sample(
            (num, self.n_features)) / torch.sqrt(alpha_sample)
        return torch.cat([omega_sample, torch.log(alpha_sample)], dim=-1)

    # calculate log(q) = log( p (param|data) )
    def q_log_prob(self, data: torch.Tensor):
        # data is T(x) [train_b, n_features + 1]
        # train_data_batch[given_data_b, n_features + 1]
        # train_data_batch[:, 0] is class label -1 or 1
        train_data_batch = self.ds_sample().to(data.device)
        assert len(data.shape) == 2
        assert data.size(1) == train_data_batch.size(1)
        assert data.size(1) == self.n_features + 1  # features + alpha

        # calculate log p_0 (params) = log p_0 (omega | alpha) + log p_0 (alpha)
        log_alpha_sample = data[:, -1].view(-1, 1)
        if self.cfg.clip_alpha is not None:
            log_alpha_sample = torch.clamp(
                log_alpha_sample, -self.cfg.clip_alpha, self.cfg.clip_alpha)
        alpha_sample = torch.exp(log_alpha_sample)
        omega_sample = data[:, :-1]
        log_p_w_cond_alp = self.normal0.log_prob(
            omega_sample * torch.sqrt(alpha_sample)).sum(dim=-1)
        log_p_alp = self.gamma0.log_prob(alpha_sample)
        log_p = log_p_alp.view(-1) + log_p_w_cond_alp
        assert len(log_p.shape) == 1

        # calculate  log p(s_i | params) sum over i in dataset size
        probas = torch.sigmoid(torch.matmul(
            data[:, :-1], train_data_batch[:, 1:].T))
        # probas = (x_bs, s_bs)
        classes = train_data_batch[:, 0].view(1, -1)
        probas = (1. - classes) / 2. + classes * probas
        probas = torch.clamp(probas, min=1e-5)
        log_probas = torch.log(probas)
        summed_log_probas = torch.mean(log_probas, dim=-1) * self.len_dataset
        assert summed_log_probas.size(0) == data.size(0)
        assert len(summed_log_probas.shape) == 1
        return log_p + summed_log_probas

    def ds_sample(self):
        try:
            data, classes = next(self._dataloader_iter)
            assert data.size(1) == self.n_features
            self.n_data_samples_drawn += 1
        except StopIteration:
            self._dataloader_iter = iter(self.train_dl)
            self.n_data_epochs_drawn += 1
            data, classes = next(self._dataloader_iter)
        batch = torch.cat([
            classes.view(-1, 1).type(torch.float32),
            data.type(torch.float32)], dim=-1)
        return batch

    def to_device(self, device):
        self.mu_gauss = deep_to(self.mu_gauss, device)
        self.gamma0 = deep_to(self.gamma0, device)
        self.normal0 = deep_to(self.normal0, device)

    @classmethod
    def instantiate(cls, merge_str=None):
        from omegaconf import OmegaConf

        default = """
        batch_size: 1024
        data_batch_size: 512
        num_workers: 0
        pin_memory: False
        mu_var: 3.0
        n_train_samples: 102400
        ds_name: "banana"
        clip_alpha: 8.0
        """
        cfg = OmegaConf.create(default)
        if merge_str is not None:
            merge_cfg = OmegaConf.create(merge_str)
            cfg = OmegaConf.merge(cfg, merge_cfg)
        data_module = cls(cfg)
        return data_module
