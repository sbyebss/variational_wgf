import torch
from src.models.loss_zoo_a import acti_js


def constraint_loss(list_of_params):
    loss_val = 0
    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val


def w2_loss(y_data, Ty, step_a):
    return (y_data - Ty).pow(2).flatten(start_dim=1).sum(-1).mean() / 2 / step_a


def get_capital_b(energy_type):
    b_dict = {
        "kl_density": capital_b_kl,
        "kl_sample": capital_b_kl,
        "js_sample": capital_b_js,
        "gan_sample": capital_b_gan,
        "gen_entropy": capital_b_gen_entropy,
    }
    return b_dict[energy_type]


def capital_b_kl(h, z, **kwargs):
    smooth = kwargs["smooth"]
    log_ratio = kwargs["log_ratio"]
    dk_formula = kwargs["dk_formula"]
    assert smooth == False or log_ratio == False
    if log_ratio or smooth:
        inside_expec = h(z).exp()
    else:
        inside_expec = h(z)
    if dk_formula:
        return inside_expec.mean().log()
    else:
        return inside_expec.mean()


def capital_b_js(h, z, **kwargs):
    activated_h_Tx = acti_js(h(z))
    return -activated_h_Tx.log().mean()


def capital_b_gan(h, z, **kwargs):
    return -(1 - torch.sigmoid(h(z))).clamp(min=1e-30).log().mean()


def capital_b_gen_entropy(h, z, **kwargs):
    log_ratio = kwargs["log_ratio"]
    smooth_h = kwargs["smooth_h"]
    assert smooth_h == False or log_ratio == False
    q_func = kwargs["q_func"]
    assert type(q_func) == float
    m = kwargs["m"]
    if log_ratio:
        return ((h(z).exp())**m).mean() * q_func**(m - 1)
    elif smooth_h:
        return ((h(z).exp() - 1)**m).mean() * q_func**(m - 1)
    else:
        return (h(z)**m).mean() * q_func**(m - 1)
