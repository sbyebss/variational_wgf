import pytest
from src.datamodules.bayesian_datamodule import BayesianDataModule


@pytest.mark.parametrize("merge_cfg", [None])
def test_bayesian(merge_cfg):
    data_module = BayesianDataModule.instantiate(merge_cfg)
    data_module.setup()
    data_module.to_device("cuda")

    p0_data = data_module.p0_sample(1000)
    data_module.q_log_prob(p0_data)
    mu_data = data_module.mu_sample(data_module.cfg.n_train_samples)
    data_module.mu_log_prob(mu_data)

    data_module.update_gamma(mu_data)
    new_data = data_module.mu_sample(data_module.cfg.n_train_samples)
    data_module.update_pk(new_data)
