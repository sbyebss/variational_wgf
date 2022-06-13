from src.callbacks.metric_zoo.ksd import KSD
import torch.distributions as TD
import torch


def test_ksd():
    dims = 2
    mix = TD.Categorical(torch.ones(10,))
    comp = TD.Independent(TD.Normal(
        torch.FloatTensor(
            10,
            dims).uniform_(-2.5, 2.5),
        torch.ones(10, dims)), 1)
    q_gmm = TD.MixtureSameFamily(mix, comp)

    mu_gauss = TD.MultivariateNormal(torch.zeros(
        dims), 16 * torch.eye(dims))
    ksd = KSD(q_gmm.log_prob, beta=0.0001)
    print(ksd(q_gmm.sample_n(10000), q_gmm.sample_n(10000), True))
    # samples = q_gmm.sample_n(1000)
    # print(ksd(samples))
    samples1 = mu_gauss.sample_n(10000)
    samples2 = mu_gauss.sample_n(10000)
    print(ksd(samples1, samples2, adjust_beta=True))


a = test_ksd()
