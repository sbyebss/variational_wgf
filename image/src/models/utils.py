import inspect
from collections import defaultdict

from torch import nn

g_caller_cnt = defaultdict(lambda: 0)


def bufcnt(cond):
    frames = inspect.stack()
    hash_info = "".join(
        [f"{cur_st.filename}{cur_st.lineno}\n" for cur_st in frames[1:]]
    )
    g_caller_cnt[hash_info] = g_caller_cnt[hash_info] * cond + cond
    return g_caller_cnt[hash_info]


def get_weights_init_fn(init_type="xavier_uniform"):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "orth":
                nn.init.orthogonal_(m.weight.data)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform(m.weight.data, 1.0)
            else:
                raise NotImplementedError("{} unknown inital type".format(init_type))
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("GroupNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    return weights_init
