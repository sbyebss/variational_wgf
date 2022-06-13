from scipy.stats import gaussian_kde
import torch


def compare_highd_h(uniform_sampler, h_net, pk_data, q_density):
    test_samples = uniform_sampler(2000)
    h_output = h_net(test_samples).detach().cpu().reshape(-1)
    pk_data_sel = pk_data[:10000]
    pk_fitted = gaussian_kde(pk_data_sel.T)
    # pk_density = diff.log_prob(test_samples).exp()
    pk_density = (torch.from_numpy(pk_fitted.logpdf(test_samples.T.detach().cpu()))).exp()
    assert h_output.shape == pk_density.shape
    matter_pos = pk_density > 0.1
    difference = h_output[matter_pos] - pk_density[matter_pos] / q_density
    if difference.shape[0] > 0:
        return difference.abs().mean(), difference.abs().max()
    else:
        return -1, -1


def log10symkl_icnn(gt_density, gt_sampler, pk_data: torch.Tensor, pk_density_fitted):
    gt_data = gt_sampler(10000)
    pk_data_sel = pk_data[:10000]
    kl1 = (gt_density(gt_data) + 1e-45).log().mean() - \
        (pk_density_fitted(gt_data) + 1e-45).log().mean()
    kl2 = (pk_density_fitted(pk_data_sel) + 1e-45).log().mean() - \
        (gt_density(pk_data_sel) + 1e-45).log().mean()

    return torch.log10(kl1 + kl2)
