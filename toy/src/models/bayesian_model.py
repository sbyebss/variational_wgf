from .kl_gmm_model import GMMModule
from src.models.data_updation import new_pk_fixed_p0_generator


class BayesianModule(GMMModule):
    def new_pk_generator(self, iterated_map):
        return new_pk_fixed_p0_generator(
            iterated_map, self.idx_pk_dist,
            self.trainer.datamodule.p0_sample, self.trainer.datamodule.train_size, self.device)
