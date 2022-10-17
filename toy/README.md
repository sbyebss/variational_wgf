# Toy examples

## Setup

```shell
pip install --no-deps -r requirements.txt
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Replace the `gpus` in `configs/trainer/default.yaml` by the GPU id you want to use. For example, if you want to use GPU 3, then write `gpus: [3]`.

## Repository structure

The part of repository highly depends on the [pytorch-lightning template](https://github.com/ashleve/lightning-hydra-template). The hyper-parameters are stored in `configs/`. The executable commmands are included in `bash/`. The visualization is realized in `notebooks/variational_wgf.ipynb`.

## Reproduce

For example:

```
python run.py -m paper=gmm_d2 logger=wandb seed=1,2,3,4,5 logger.wandb.tags=\["reproduce"\] mode=exp
```

See the [bash](bash) for more reproducing commands.

See the [config](configs/experiment) for more experiment configs.
