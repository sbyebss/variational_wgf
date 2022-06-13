import torch
from src.models.loss_zoo_a import h_loss_in_kl_cap_a, h_loss_in_gen_entropy_cap_a
from src.models.loss_zoo_b import capital_b_kl, capital_b_gen_entropy
from src.callbacks.metric_zoo.ou_symkl import kl_gaussian, ou_exact_mean_cov, extract_param_mu_q


def kl_variational_func(pk_samples, mu_gauss, h_net, log_q):
    mu_samples = mu_gauss.sample_n(pk_samples.shape[0])
    logh = h_loss_in_kl_cap_a(pk_samples, h_net, log_ratio=False, smooth=True)
    logmu = mu_gauss.log_prob(pk_samples)
    logq = log_q(pk_samples)
    h_mu = capital_b_kl(h_net, mu_samples, smooth=True, log_ratio=False, dk_formula=False)
    value = 1 + (logh + logmu - logq) - (h_mu - 1)
    assert value.shape == (pk_samples.shape[0],)
    return value.mean()


def ou_exact_func(t_now, q_gmm, verbose=False):
    mean_q, cov_q = q_gmm.mean, q_gmm.covariance_matrix
    mean_q = mean_q.reshape(-1, 1)
    mean_now, cov_now, _ = ou_exact_mean_cov(t_now, mean_q, cov_q)
    inv_cov_q = torch.inverse(cov_q)

    kl_gt_q = kl_gaussian(mean_now, cov_now, mean_q, cov_q, inv_cov_q)
    if verbose:
        print("mean = ", mean_now, mean_q)
        print("cov = ", cov_now, cov_q)
        print("calculate the entropy", "kl_gt_q=", kl_gt_q)
    return kl_gt_q


def ou_variation_func(mu, q_gmm, verbose=False):
    mean_mu, cov_mu, mean_q, cov_q = extract_param_mu_q(mu, q_gmm)
    inv_cov_q = torch.inverse(cov_q)
    kl_pk_q = kl_gaussian(mean_mu, cov_mu, mean_q, cov_q, inv_cov_q)
    if verbose:
        print("mean = ", mean_mu, mean_q)
        print("cov = ", cov_mu, cov_q)
        print("calculate the entropy", "kl_pk_q=", kl_pk_q)
    return kl_pk_q


def gen_entropy_variational_func(pk_samples, q_uniform, q_density, h_net, m):
    q_samples = q_uniform.sample_n(pk_samples.shape[0])
    loss_a = h_loss_in_gen_entropy_cap_a(
        pk_samples, h_net, q_density, m)
    loss_b = capital_b_gen_entropy(
        h_net, q_samples, smooth_h=False,
        log_ratio=False, q_func=q_density, m=m)
    value = loss_a - loss_b
    assert value.shape == ()
    return value


def gen_entropy_exact_func(gt_sampler, gt_denstiy, porous_m, num_samples=50000):
    # \int P^m is equal to E_P [P^{m-1}]
    gt_samples = gt_sampler(num_samples)
    pk_density = gt_denstiy(gt_samples)
    return (pk_density**(porous_m - 1)).mean() / (porous_m - 1)
