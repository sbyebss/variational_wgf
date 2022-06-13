import pytest
from src.datamodules.gmm_datamodule import GMMDataModule
from sklearn.mixture import GaussianMixture as GM
# from src.viz.points import highd_kde_scatter
gmm_cfg_mode1 = {
    "num_gmm_component": 1
}
gmm_cfg_mode2 = {
    "num_gmm_component": 10
}


@pytest.mark.parametrize("merge_cfg", [None, gmm_cfg_mode1, gmm_cfg_mode2])
def test_gmm(merge_cfg):
    data_module = GMMDataModule.instantiate(merge_cfg)
    data_module.setup()
    data_module.to_device("cuda")

    q_data = data_module.q_gmm.sample_n(data_module.cfg.n_train_samples)
    num_fitted = 10000
    pk_fitted = GM(n_components=merge_cfg["num_gmm_component"],
                   covariance_type='full').fit(q_data[:num_fitted].cpu())
    fitted_density = pk_fitted.score_samples(q_data[:num_fitted].cpu())
    gt_density = data_module.q_log_prob(q_data[:num_fitted])
    print((fitted_density - gt_density.cpu().numpy()).mean())
    assert (fitted_density - gt_density.cpu().numpy()).mean() < 2
    import scipy
    pk_fit_scipy = scipy.stats.gaussian_kde(q_data[:50000].cpu().T)
    scipy_density = pk_fit_scipy.logpdf(q_data[:50000].cpu().T)
    gt_density = data_module.q_log_prob(q_data[:50000])
    assert (scipy_density - gt_density.cpu().numpy()).mean() < 2

    mu_data = data_module.mu_sample(data_module.cfg.n_train_samples)
    data_module.q_log_prob(q_data)
    data_module.mu_log_prob(mu_data)

    data_module.update_gamma(q_data)
    new_data = data_module.mu_sample(data_module.cfg.n_train_samples)
    data_module.update_pk(new_data)
    # plot q_data,mu_data
    # highd_kde_scatter(q_data, f'data/tmp/q_{data_module.cfg.num_gmm_component}.png', 'q')
    # highd_kde_scatter(mu_data, f'data/tmp/mu.png', 'mu')
