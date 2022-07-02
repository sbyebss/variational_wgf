from abc import ABC, abstractmethod

from omegaconf import DictConfig
from pytorch_lightning import LightningModule

# pylint: disable=too-many-ancestors,arguments-differ,attribute-defined-outside-init,unused-argument,too-many-instance-attributes, abstract-method


class BaseModel(LightningModule, ABC):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.x_shape = tuple(cfg.x_shape)
        self.z_shape = tuple(cfg.z_shape)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.instantiate()

    def on_pretrain_routine_start(self) -> None:
        self.data_transform_fn = self.trainer.datamodule.data_transform
        self.inverse_data_transform_fn = self.trainer.datamodule.inverse_data_transform

    @abstractmethod
    def instantiate(self):
        ...

    @abstractmethod
    def forward(self, z_vars=None, num_sample=None):
        ...

    @abstractmethod
    def sample_n(self, num_sample):
        ...

    @abstractmethod
    def logp(self, x):
        ...
