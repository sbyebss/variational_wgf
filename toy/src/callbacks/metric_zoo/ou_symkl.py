import torch
import torch.distributions as TD


def extract_param_mu_q(mu, q_gmm):
    mean_mu, cov_mu = mu.mean, mu.covariance_matrix
    mean_q, cov_q = q_gmm.mean, q_gmm.covariance_matrix
    mean_mu = mean_mu.reshape(-1, 1)
    mean_q = mean_q.reshape(-1, 1)
    return mean_mu, cov_mu, mean_q, cov_q


def ou_exact_mean_cov(t_now, mean_q, cov_q):
    # Warning: This assume P_0 is N(0,I), otherwise not correct
    dim = cov_q.shape[0]
    inv_cov_q = torch.inverse(cov_q)

    exp2m = torch.matrix_exp(-2 * inv_cov_q * t_now)
    expm = torch.matrix_exp(-inv_cov_q * t_now)
    identity = torch.eye(dim).to(mean_q.device)
    mean_now = (identity - expm) @ mean_q
    cov_now = cov_q @ (identity - exp2m) + exp2m
    inv_cov_now = torch.inverse(cov_now)

    return mean_now, cov_now, inv_cov_now


def log10symkl_gauss_with_mean_cov(q_gmm, mu, t_now, verbose=False):
    mean_mu, cov_mu, mean_q, cov_q = extract_param_mu_q(mu, q_gmm)
    mean_now, cov_now, inv_cov_now = ou_exact_mean_cov(t_now, mean_q, cov_q)
    inv_cov_mu = torch.inverse(cov_mu)
    kl_mu_gt = kl_gaussian(mean_mu, cov_mu, mean_now, cov_now, inv_cov_now)
    kl_gt_mu = kl_gaussian(mean_now, cov_now, mean_mu, cov_mu, inv_cov_mu)
    if verbose:
        print("current_time=", t_now)
        print("mean = ", mean_now, mean_mu)
        print("cov = ", cov_now, cov_mu)
        print("calculate the log10symKL", "kl_mu_gt=", kl_mu_gt, "kl_gt_mu=", kl_gt_mu)
    return torch.log10(kl_mu_gt + kl_gt_mu)


def log10symkl_gauss_mcmc(q_gmm, mu, pk_data, t_now, verbose=False):
    mean_q, cov_q = q_gmm.mean, q_gmm.covariance_matrix
    assert torch.all(torch.eq(pk_data.mean(axis=0), mu.loc.cpu()))
    assert torch.all(torch.eq(torch.cov(pk_data.T), mu.covariance_matrix.cpu()))
    mean_q = mean_q.reshape(-1, 1)
    mean_now, cov_now, _ = ou_exact_mean_cov(t_now, mean_q, cov_q)

    if torch.all(mean_now == mu.loc) and torch.all(cov_now == mu.covariance_matrix):
        print("current mean and cov are equal to ground truth")
        return -10000
    gt_gauss = TD.MultivariateNormal(mean_now.reshape(-1), cov_now)
    q_data = gt_gauss.sample((10000,))
    pk_data_sel = pk_data[:10000].to(mean_q.device)
    kl1 = gt_gauss.log_prob(q_data).mean() - mu.log_prob(q_data).mean()
    kl2 = mu.log_prob(pk_data_sel).mean() - gt_gauss.log_prob(pk_data_sel).mean()
    if verbose:
        print("current_time=", t_now)
        print("mean = ", mean_now, mu.loc)
        print("cov = ", cov_now, mu.covariance_matrix)
        print("log10symKL", "kl1=", kl1, "kl2=", kl2)
    return torch.log10(kl1 + kl2)


def kl_gaussian(mean_p, cov_p, mean_q, cov_q, inv_cov_q):
    dim = cov_p.shape[0]
    part1 = torch.log(torch.det(cov_q) / torch.det(cov_p)) - dim
    part2 = (mean_p - mean_q).T @ inv_cov_q @ (mean_p - mean_q)
    part3 = torch.trace(inv_cov_q @ cov_p)
    kl_between_p_q = (part1 + part2 + part3) / 2
    # print("inv_cov_q=", inv_cov_q)
    # print("part1=", part1, "part2=", part2, "part3=", part3)
    return torch.clamp(kl_between_p_q, min=1e-10)
