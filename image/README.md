## Setup

The repo heavily depends on [jam](https://github.com/qsh-zh/jam), a versatile toolbox developed by [Qsh.zh](https://github.com/qsh-zh) and [jam-hlt](https://github.com/qsh-zh/jam), a decent deep leanring project template. [⭐️](https://github.com/qsh-zh/jam) if you like them.

_[poetry](https://python-poetry.org/)_ (**Recommended**)

```shell
curl -fsS -o /tmp/get-poetry.py https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py
python3 /tmp/get-poetry.py -y --no-modify-path
export PATH=$HOME/.poetry/bin:$PATH
poetry shell
poetry install
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

_pip_

```shell
pip install .
```

## Reproduce

```shell
python run.py experiment=mnist
python run.py experiment=cifar
```
