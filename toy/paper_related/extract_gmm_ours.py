import jammy.io as io

pk_list = {}
target_list = {}

for dim in [2, 4, 8, 17, 24, 32, 64, 128]:
    target_list[dim] = {}
    pk_list[dim] = {}
    for idx_exp in range(5):
        data_path = f"logs/papers/gmm/multirun/{dim}/{idx_exp+1}"
        target_dist = io.load(data_path + f"/target_dist.pth")

        pk_data = io.load(data_path + f"/pk_data_40.pth")
        target_list[dim][idx_exp] = target_dist
        pk_list[dim][idx_exp] = pk_data

io.fs.dump(f"paper_plots/ours_gmm_data.pth",
           {"pk_data": pk_list, "target_dist": target_list})
