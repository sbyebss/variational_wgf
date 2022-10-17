# d3_recover
python run.py experiment=porous_media mode=paper datamodule.input_dim=3 name="porous" datamodule.p0_bound=0.4 model.N_outer_ITERS=2 logger=wandb  logger.wandb.tags=\["check_functional"\]

# d6_recover
python run.py experiment=porous_media mode=paper datamodule.input_dim=6 name="porous" datamodule.p0_bound=0.6 model.N_outer_ITERS=1 model.h_net.num_layer=3 logger=wandb logger.wandb.tags=\["check_functional"\]
