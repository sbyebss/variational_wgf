import numpy as np
import torch as th
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import src.datamodules.datasets.points_dataset as psd

# pylint: disable=abstract-method


class PointsDataModule(LightningDataModule):
    def __init__(
        self,
        points_name: str,
        noise: float = 0.001,
        data_std: float = 1.0,
        batch_size: int = 128,
        num_batch: int = 100,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        # self.dims is returned when you call datamodule.size()
        self.dims = (2,)
        self.points_name = points_name
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.pin_memory = pin_memory

        self.train_set = psd.PointsDataSet(points_name, batch_size * num_batch, noise)
        self.train_set.normalize(data_std)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def get_test_samples(self, num_sample=5000):
        data = self.train_set.data
        return th.from_numpy(
            data[np.random.randint(data.shape[0], size=num_sample)]
        ).float()

    def data_transform(self, x):  # pylint: disable=no-self-use
        return x.float()

    def inverse_data_transform(self, x):  # pylint: disable=no-self-use
        return x

    @property
    def bpd_offset(self):
        return 8.0
