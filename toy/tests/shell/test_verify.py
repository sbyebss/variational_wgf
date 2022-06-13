from tests.helpers.run_command import run_command
from tests.helpers.runif import RunIf


@RunIf(min_gpus=1)
def test_verify():
    command = [
        "run.py",
        "paper=gmm_d2",
        "name=gmm_d2_verify",
        "mode=exp",
        "++trainer.max_steps=10",
        "++trainer.log_every_n_steps=1",
        "logger.wandb.tags=[verify, test]"
    ]
    run_command(command)


def test_baseline():
    command = [
        "run.py",
        "paper=gmm_d2",
        "name=gmm_d2",
        "mode=exp",
        "++trainer.max_steps=10",
        "++trainer.log_every_n_steps=1",
        "logger.wandb.tags=[verify,baseline]"
    ]
    run_command(command)
