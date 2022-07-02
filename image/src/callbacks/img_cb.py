import pytorch_lightning as pl  # pylint: disable=unused-import
import torch as th
from pytorch_lightning import Callback

from src.viz.img import save_tensor_imgs


class ImgViz(Callback):
    def on_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.global_step % 500 == 0:
            with th.no_grad():
                fake_img = pl_module(num_sample=100)
                save_tensor_imgs(
                    pl_module.inverse_data_transform_fn(fake_img),
                    10,
                    pl_module.global_step,
                    "fake",
                )
