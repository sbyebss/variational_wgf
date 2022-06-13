import jammy.io as io
# from src.callbacks.metric_zoo.ksd import KSD
# import numpy as np

tmp_a = [2, 4, 8, 15, 24, 32]
tmp_a.reverse()
dims_all = [tmp_a, [15, 24, 32]]
methods = ["full", "appro"]
# dims_all = [[15, 24, 32]]
# methods = ["appro"]
pk_list = {}
target_list = {}

for dims, method in zip(dims_all, methods):
    # stein_result = np.zeros([len(dims), 5])
    idx_dim = 0
    for dim in dims:
        # dim_beta = dim if dim != 15 else 17
        # beta = io.load(
        #     f"paper_plots/beta_values/beta_{dim_beta}.pth")
        target_list[dim] = {}
        pk_list[dim] = {}
        for idx_exp in range(5):
            target_dist = io.load(
                f"../PytorchSource/results/conv_comp_dim_{dim}/{method}/model-conv_comp_dim_{dim}-ICNN_jko-dim={dim}-n_exp={idx_exp+1}.pth").target_distrib
            # ksd = KSD(target_dist.log_prob, beta=beta)
            pk_data = io.load(
                f"../PytorchSource/results/conv_comp_dim_{dim}/{method}/pk_data_exp{idx_exp+1}.pth")[-1]
            target_list[dim][idx_exp] = target_dist
            pk_list[dim][idx_exp] = pk_data
            # tmp_result = 0.
            # for idx_repeat in range(10):
            #     tmp_result += ksd(pk_data[3000 * idx_repeat:3000 * (idx_repeat + 1)])
            # stein_result[idx_dim, idx_exp] = tmp_result / 10.
        idx_dim += 1
    # print(stein_result)
    io.fs.dump(f"paper_plots/jko_icnn_{method}_gmm_data.pth",
               {"pk_data": pk_list, "target_dist": target_list})
    # io.fs.dump(f"paper_plots/jko_icnn_{method}_ksd.npy", stein_result)
