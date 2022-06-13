from typing import Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.datamodules.datasets.image_dataset import get_img_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, x):
    if config.uniform_dequantization:
        x = x / 256.0 * 255.0 + torch.rand_like(x) / 256.0
    if config.gaussian_dequantization:
        x = x + torch.randn_like(x) * 0.01

    if config.rescaled:
        x = 2 * x - 1.0
    elif config.logit_transform:
        x = logit_transform(x)

    if config.image_mean is not None and config.image_std is not None:
        return (
            x - torch.FloatTensor(config.image_mean).to(x.device)[:, None, None]
        ) / torch.FloatTensor(config.image_std).to(x.device)[:, None, None]
    return x


def inverse_data_transform(config, x):
    if config.image_mean is not None and config.image_std is not None:
        x = (
            x * torch.FloatTensor(config.image_std).to(x.device)[:, None, None]
            + torch.FloatTensor(config.image_mean).to(x.device)[:, None, None]
        )

    if config.logit_transform:
        x = torch.sigmoid(x)
    elif config.rescaled:
        x = (x + 1.0) / 2.0

    return x


class ImgDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # self.dims is returned when you call datamodule.size()
        self.dims = (cfg.channel, cfg.image_size, cfg.image_size)
        self.data_train: Optional[Dataset] = None

    @property
    def train_size(self):
        return len(self.data_train)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        get_img_dataset(self.cfg)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately
        when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called
        before trainer.fit()` or `trainer.test()`."""
        self.data_train = get_img_dataset(self.cfg)
        self.pk_data = torch.randn(
            len(self.data_train), self.cfg.channel, self.cfg.image_size,
            self.cfg.image_size)

    def train_dataloader(self):
        loader_pk = DataLoader(
            self.pk_data, batch_size=self.cfg.dl.batch_size,
            shuffle=True, num_workers=self.cfg.dl.num_workers)
        loader_img = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.dl.batch_size,
            num_workers=self.cfg.dl.num_workers,
            pin_memory=self.cfg.dl.pin_memory,
            shuffle=True,
        )
        return [loader_pk, loader_img]

    def data_transform(self, feed_dict):
        if isinstance(feed_dict, list):
            feed_dict = feed_dict[0]
        feed_dict = feed_dict.float()
        return data_transform(self.cfg, feed_dict)

    def inverse_data_transform(self, x):
        return inverse_data_transform(self.cfg, x)

    def update_pk(self, new_pk_data):
        self.pk_data = new_pk_data
