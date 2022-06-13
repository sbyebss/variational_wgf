from matplotlib import pyplot as plt
import pytest
from src.datamodules.gen_energy_datamodule import BarenblattDataModule
import torch
from src.viz.points import compare_highd_kde_scatter
bar_cfg_mode1 = {
    "input_dim": 10
}
bar_cfg_mode2 = {
    "input_dim": 2
}


@pytest.mark.parametrize("merge_cfg", [None, bar_cfg_mode1, bar_cfg_mode2])
def test_baren(merge_cfg):
    data_module = BarenblattDataModule.instantiate(merge_cfg)
    # data_module.setup()
    data_module.to_device("cuda")
    # for _ in range(50):
    #     p0_data = data_module.p0_sample(10000)
    #     assert p0_data.abs().max().item() <= 0.25
    p0_data = data_module.p0_sample(40000)
    print(p0_data.max())

    def gt_sampler(num):
        return data_module.gt_sample(num, data_module.cfg.t0)

    compare_highd_kde_scatter(
        p0_data, gt_sampler, f"trash/test.png", plot_size=1, levels=5)
    q_data = data_module.q_sample(4000)
    compare_highd_kde_scatter(
        q_data, gt_sampler, f"trash/test_q.png", plot_size=1, levels=5)
    prop_data = data_module.prop.sample_n(4000)
    compare_highd_kde_scatter(
        prop_data, gt_sampler, f"trash/test_prop.png", plot_size=1, levels=5)
    # x_array = torch.linspace(-1, 1, 1000).reshape(-1, 1)
    # density_p0 = data_module.density_p0_unnml(x_array)
    # density_prop = data_module.prop.log_prob(x_array).exp()

    # fig, ax = plt.subplots(figsize=(8, 4.6), dpi=80)
    # # ax.hist(p0_data.numpy(), bins=1000, range=[-1, 1], density=True, alpha=0.5)
    # ax.plot(x_array.view(-1), density_p0 / density_p0.sum() * 1000 / 2)
    # ax.plot(x_array.view(-1), density_p0 / density_p0.sum() * 1000 / 2 / density_prop)
    # plt.grid(which='major')
    # plt.minorticks_on()
    # plt.tight_layout()
    # fig.savefig("trash/tst.png", bbox_inches='tight', dpi=200)
