import torch
from sklearn.mixture import GaussianMixture as GM
from scipy.stats import gaussian_kde
from src.utils.th_utils import deep_to


def log10symkl_gmm_icnn(q_gmm: torch.distributions.MixtureSameFamily, diff, log_pk_fitted):
    if log_pk_fitted == None:
        return -1
    q_gmm_cpu = deep_to(q_gmm, 'cpu')
    q_data = q_gmm_cpu.sample_n(10000)
    p0_data = diff.sample_init(10000)
    entropy, pk_data = diff.mc_entropy(p0_data, return_X_transformed=True)
    kl1 = q_gmm_cpu.log_prob(q_data).mean() - log_pk_fitted(q_data).mean()
    kl2 = entropy.detach().cpu() - q_gmm_cpu.log_prob(pk_data.detach().cpu()).mean()
    return torch.log10(kl1 + kl2)


def log10symkl_gmm_kde(q_gmm: torch.distributions.MixtureSameFamily, pk_data: torch.Tensor, num_comp=20, kde_method="GM"):
    q_gmm_cpu = deep_to(q_gmm, 'cpu')
    if kde_method == "GM":
        q_data = q_gmm_cpu.sample_n(50000)
        pk_data_sel = pk_data[:50000]
        pk_fitted = GM(n_components=num_comp, covariance_type='full').fit(pk_data_sel)

        kl1 = q_gmm_cpu.log_prob(q_data).mean() - pk_fitted.score_samples(q_data).mean()
        kl2 = pk_fitted.score_samples(pk_data_sel).mean() - q_gmm_cpu.log_prob(pk_data_sel).mean()
    elif kde_method == "gaussian_kde":
        q_data = q_gmm_cpu.sample_n(10000)
        pk_data_sel = pk_data[:10000]
        pk_fitted = gaussian_kde(pk_data_sel.T)

        kl1 = q_gmm_cpu.log_prob(q_data).mean() - pk_fitted.logpdf(q_data.T).mean()
        kl2 = pk_fitted.logpdf(pk_data_sel.T).mean() - q_gmm_cpu.log_prob(pk_data_sel).mean()
    return torch.log10(kl1 + kl2)
