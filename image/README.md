## Setup

The repo heavily depends on [jam](https://github.com/qsh-zh/jam), a versatile toolbox developed by [Qsh.zh](https://github.com/qsh-zh) and [jam-hlt](https://github.com/qsh-zh/jam), a decent deep leanring project template. [⭐️](https://github.com/qsh-zh/jam) if you like them.

*[poetry](https://python-poetry.org/)* (**Recommended**)
```shell
curl -fsS -o /tmp/get-poetry.py https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py
python3 /tmp/get-poetry.py -y --no-modify-path
export PATH=$HOME/.poetry/bin:$PATH
poetry shell
poetry install
```

*pip*
```shell
pip install .
```

## Reproduce

```shell
python run.py experiment=mnist
python run.py experiment=cifar
```
