import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from .wandb_fig import wandb_img
import torch


def compare_highd_kde_scatter(data, q_sampler, fig_path, plot_size=12, levels=10, figsize=(8, 3.5)):
    pk_cpu = data.detach().cpu()
    pca = PCA(n_components=2)
    target_data = q_sampler(4000).detach().cpu()
    pca.fit_transform(target_data)
    gen_data_pca = pca.transform(pk_cpu[:4000])
    target_pca = pca.transform(target_data)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, facecolor='w')
    sns.kdeplot(x=target_pca[:, 0], y=target_pca[:, 1], ax=ax[0],
                color='mediumaquamarine', linewidths=3.0, levels=levels, alpha=0.7)
    xlims = (-plot_size, plot_size)
    ylims = (-plot_size, plot_size)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_title('Stationary measure', fontsize=18)
    ax[0].scatter(target_pca[:, 0], target_pca[:, 1], color='darkslategray', alpha=0.1)
    ax[0].grid()
    sns.kdeplot(x=gen_data_pca[:, 0], y=gen_data_pca[:, 1], ax=ax[1],
                color='darkturquoise', linewidths=3.0, levels=levels, alpha=0.7)
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    ax[1].set_title('Fitted measure (ours)', fontsize=18)
    ax[1].scatter(gen_data_pca[:, 0], gen_data_pca[:, 1], color='darkslategray', alpha=0.1,)
    ax[1].grid()
    fig.savefig(fig_path, bbox_inches='tight', dpi=200)
    fig.suptitle(fig_path)
    plt.show()
    plt.close()
    wandb_img("kde-scatter", fig_path, fig_path)


def plot_porous_1d(data, density_fit, density_pt, density_q, h_net, plot_bound, epoch):
    x_plot = torch.linspace(-plot_bound, plot_bound,
                            100).reshape(-1, 1)
    Pk_hist = plt.hist(
        data.numpy(), bins=100,
        range=[-plot_bound, plot_bound], density=True, color='C0', alpha=0.5)[0]
    gt_rho = density_pt(x_plot)
    gt_rho /= gt_rho.sum()
    gt_rho /= (2 * plot_bound / 100)
    plt.plot(x_plot, gt_rho, c='C1', label='exact measure')
    if density_fit != None:
        fitted_rho = density_fit(x_plot)
        fitted_rho /= fitted_rho.sum()
        fitted_rho /= (2 * plot_bound / 100)
        plt.plot(x_plot, fitted_rho, c='C2', label='fitted measure')

    density_path = f"epoch{epoch}_density.png"
    plt.savefig(density_path, bbox_inches='tight', dpi=200)
    plt.close()
    wandb_img("density-comparison", density_path, density_path)

    x_axis = x_plot.reshape(-1, 1).cuda()
    plt.plot(x_plot, torch.clamp(h_net(x_axis).detach().cpu(), min=-1, max=10), c='C0')
    plt.plot(x_plot, Pk_hist / density_q, c='C1')

    h_path = f"epoch{epoch}_h.png"
    plt.savefig(h_path, bbox_inches='tight', dpi=200)
    plt.close()
    wandb_img("h-comparison", h_path, h_path)
