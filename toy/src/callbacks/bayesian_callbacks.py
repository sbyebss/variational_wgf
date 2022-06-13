import pytorch_lightning as pl
from src.callbacks.ou_callbacks import DataCb
from src.logger.jam_wandb import prefix_metrics_keys
import jammy.io as io
import numpy as np
import numpy.matlib as nm


def dataset2numpy(dataset):
    X = np.stack([dataset[i][0] for i in range(len(dataset))])
    y = np.stack([dataset[i][1] for i in range(len(dataset))])
    return X, y


class Accuracy_LogLikelihood_Cb(DataCb):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.epoch % pl_module.cfg.epochs_per_val == 0:
            pl_module.iterate_dataloader()
            # theta: [n_param, n_feature]
            # X_test: [n_test, n_feature]
            # y_test: [n_test]
            X_test, y_test = dataset2numpy(trainer.datamodule.test_ds)
            theta = trainer.datamodule.pk_data[:4096].to(pl_module.device)
            theta = (theta[:, :-1]).detach().cpu()
            n_param, n_test = theta.shape[0], len(y_test)

            prob = np.zeros([n_test, n_param])  # the probability to produce correct prediction
            for t in range(n_param):
                coff = np.multiply(
                    y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
                prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))

            prob = np.mean(prob, axis=1)
            acc = np.mean(prob > 0.5)
            llh = np.mean(np.log(prob))

            io.fs.dump(f'acc_{pl_module.idx_pk_dist}.pth', acc)
            io.fs.dump(f'llh_{pl_module.idx_pk_dist}.pth', llh)

            log_info = prefix_metrics_keys({
                "current_k": pl_module.idx_pk_dist,
                "acc": acc,
                "llh": llh
            },
                "bayesian")
            pl_module.log_dict(log_info)
