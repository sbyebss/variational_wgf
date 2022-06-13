import matplotlib.pyplot as plt
import torch
import wandb
import jammy.image.imgio as imgio
from torchvision.utils import make_grid

plt.switch_backend("agg")


def tensor2imgnd(tensor, n_rows, n_cols=0):  # pylint: disable=unused-argument
    grid = make_grid(tensor, n_rows)
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return ndarr


def wandb_write_ndimg(img, cur_cnt, naming):
    imgio.imwrite(f"{naming}_ep{cur_cnt}.png", img)
    if wandb.run is not None:
        wandb.log(
            {naming: wandb.Image(f"{naming}_ep{cur_cnt}.png", caption=f"{naming}_ep{cur_cnt}")}
        )


def save_tensor_imgs(x, num_grid, num_epoch, fname):
    img = tensor2imgnd(x, num_grid, num_grid)
    wandb_write_ndimg(img, num_epoch, fname)
