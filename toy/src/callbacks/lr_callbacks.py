from pytorch_lightning import Callback
import pytorch_lightning as pl
from src.logger.jam_wandb import prefix_metrics_keys


class Lr_Cb(Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        schs = pl_module.lr_schedulers()
        if schs != None:
            sch1, sch2 = schs
            sch1.step()
            sch2.step()
        if hasattr(pl_module.cfg, "step_a_schedule_epoch"):
            if pl_module.epoch % pl_module.cfg.step_a_schedule_epoch == 0:
                pl_module.cfg.step_a *= pl_module.cfg.step_a_schedule_scale
            log_info = prefix_metrics_keys({
                "step_a": pl_module.cfg.step_a
            },
                "step_a")
            pl_module.log_dict(log_info)
