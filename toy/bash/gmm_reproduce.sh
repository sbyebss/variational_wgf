# paper reproduce
python run.py -m paper=gmm_d2 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp

python run.py -m paper=gmm_d4 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp

python run.py -m paper=gmm_d8 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp

python run.py -m paper=gmm_d17 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp

python run.py -m paper=gmm_d24 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp

python run.py -m paper=gmm_d32 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp

python run.py -m paper=gmm_d64 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] 

python run.py -m paper=gmm_d128 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] 

#! test

python run.py paper=gmm_d2 seed=1 mode=evaluation skip_train=True test_after_training=True

# gmm_d2,gmm_d4,gmm_d8,gmm_d17,gmm_d24,gmm_d32,gmm_d64,gmm_d128
# seed=1,2,3,4,5 
python run.py -m paper=gmm_d128,gmm_d64,gmm_d2,gmm_d4,gmm_d8,gmm_d17,gmm_d24,gmm_d32 seed=1,2,3,4,5 mode=evaluation skip_train=True test_after_training=True