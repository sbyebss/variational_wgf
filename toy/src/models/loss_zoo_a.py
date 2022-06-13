import torch


def h_loss_in_kl_cap_a(Tx, h_net, log_ratio=False, smooth=False):
    if log_ratio:
        h_loss = h_net(Tx).mean()
    elif smooth:
        h_loss = torch.clamp((h_net(Tx)).exp() - 1, min=1e-5).log().mean()
    else:
        h_loss = torch.clamp(h_net(Tx), min=1e-10).log().mean()
    return h_loss


def get_capital_a_init(energy_type):
    a_dict = {
        "kl_density": capital_a_kl_density_p0,
        "gen_entropy": capital_a_gen_entropy_p0,
    }
    assert energy_type in a_dict
    return a_dict[energy_type]


def capital_a_kl_density_p0(Tx, **kwargs):
    log_p0_func = kwargs["log_p0_func"]
    log_q_func = kwargs["log_q_func"]
    loss1 = log_p0_func(Tx).mean()
    loss2 = -log_q_func(Tx).mean()
    info = {"log_p0": loss1, "neg_log_qt": loss2}
    return loss1 + loss2, info


def capital_a_gen_entropy_p0(Tx, **kwargs):
    p0_func = kwargs["p0_func"]
    m = kwargs["m"]
    loss = (m / (m - 1) * (p0_func(Tx))**(m - 1)).mean()
    info = {"a_loss": loss}
    return loss, info


def get_capital_a(energy_type):
    a_dict = {
        "kl_density": capital_a_kl_density,
        "kl_sample": capital_a_kl_sample,
        "gen_entropy": capital_a_gen_entropy,
        "js_sample": capital_a_js_sample,
        "gan_sample": capital_a_gan_sample,
    }
    assert energy_type in a_dict
    return a_dict[energy_type]


def capital_a_kl_density(Tx, h_net, **kwargs):
    # Tx: [batch_size, dim]
    smooth = kwargs["smooth"]
    log_ratio = kwargs["log_ratio"]
    log_gamma_func = kwargs["log_gamma_func"]
    log_q_func = kwargs["log_q_func"]
    opt = kwargs["opt"]
    assert smooth == False or log_ratio == False

    if opt == "map_t":
        loss1 = h_loss_in_kl_cap_a(Tx, h_net, log_ratio, smooth)
        loss2 = log_gamma_func(Tx).mean()
        loss3 = -log_q_func(Tx).mean()
        info = {
            "loss_ht_in_A": loss1,
            "log_mut": loss2,
            "neg_log_qt": loss3
        }

        return loss1 + loss2 + loss3, info
    elif opt == "h_net":
        loss1 = h_loss_in_kl_cap_a(Tx, h_net, log_ratio, smooth)
        info = {
            "loss_ht_in_A": loss1,
        }
        return loss1, info
    else:
        raise NotImplementedError()


def capital_a_kl_sample(Tx, h_net, **kwargs):
    smooth = kwargs["smooth"]
    log_ratio = kwargs["log_ratio"]
    assert smooth == False or log_ratio == False
    loss1 = h_loss_in_kl_cap_a(Tx, h_net, log_ratio, smooth)
    info = {"loss_ht_in_A": loss1}
    return loss1, info


def acti_js(x):
    # x is [b, ]
    assert x.shape == torch.Size([x.shape[0], ])
    return 2 * torch.exp(-x) / (1 + torch.exp(-x))


def capital_a_js_sample(Tx, h_net, **kwargs):
    activated_h_Tx = acti_js(h_net(Tx))
    loss1 = (2 - activated_h_Tx).log().mean()
    info = {
        "a_loss": loss1
    }
    return loss1, info


def capital_a_gan_sample(Tx, h_net, **kwargs):
    loss1 = (torch.sigmoid(h_net(Tx))).clamp(min=1e-30).log().mean()
    info = {"a_loss": loss1}
    return loss1, info


def h_loss_in_gen_entropy_cap_a(Tx, h_net, q_func, m, log_ratio=False, smooth=False):
    assert smooth == False or log_ratio == False
    if log_ratio:
        h_loss = q_func**(m - 1) * m / (m - 1) * (h_net(Tx).exp())**(m - 1)
    elif smooth:
        h_loss = q_func**(m - 1) * m / (m - 1) * (h_net(Tx).exp() - 1)**(m - 1)
    else:
        h_loss = q_func**(m - 1) * m / (m - 1) * (h_net(Tx))**(m - 1)
    return h_loss.mean()


def capital_a_gen_entropy(Tx, h_net, **kwargs):
    log_ratio = kwargs["log_ratio"]
    smooth_h = kwargs["smooth_h"]
    m = kwargs["m"]
    q_func = kwargs["q_func"]
    assert type(q_func) == float
    loss = h_loss_in_gen_entropy_cap_a(
        Tx, h_net, q_func, m, log_ratio=log_ratio, smooth=smooth_h)

    info = {"a_loss": loss}
    return loss, info
