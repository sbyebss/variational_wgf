# Toy examples

## Setup

```shell
pip install --no-deps -r requirements.txt
```

## Repository structure

The part of repository highly depends on the [pytorch-lightning template](https://github.com/ashleve/lightning-hydra-template). The hyper-parameters are stored in `configs/`. The executable commmands are included in `bash/`. The visualization is realized in `notebooks/variational_wgf.ipynb`.

### Reproduce

```
python run.py -m paper=gmm_d2 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp
```

See the [bash](bash) for more reproducing commands.

See the [config](configs/experiment) for more experiment configs.
