import torch


def new_pk_gmm_generator(iterated_map, current_k, p0, num, path='./'):
    # p0 is a torch.distribution subclass
    batch_num = 50
    batch_size = int(num / batch_num)
    p0_device = p0.sample_n(2).device
    P_total = torch.randn([num, p0.sample_n(2).shape[1]])
    for idx_k in range(1, current_k + 1):
        try:
            iterated_map.load_state_dict(torch.load(
                path + f'map_{idx_k}.pth'))
        except:
            iterated_map.load_state_dict(torch.load(
                path + f'map_{idx_k}.pt'))
        for idx_batch in range(batch_num):
            if idx_k > 1:
                P_iterate = P_total[
                    batch_size * idx_batch:batch_size *
                    (idx_batch + 1)].to(p0_device)
            else:
                P_iterate = p0.sample_n(batch_size)
            P_total[batch_size * idx_batch:batch_size *
                    (idx_batch + 1)] = map_forward(iterated_map, P_iterate, sampling=True).detach()
        P_total = P_total.detach()
    return P_total


def map_forward(map_t, data, sampling=False):
    if map_t.is_icnn:
        data.requires_grad_(True)
        if sampling:
            return map_t.push_nograd(data)
        else:
            return map_t.push(data)
    else:
        return map_t(data)


def new_pk_fixed_p0_generator(iterated_map, current_k, p0_sampler, num, device):
    batch_num = 20
    batch_size = int(num / batch_num)
    P_total = torch.randn([num, p0_sampler(2).shape[1]])
    for idx_k in range(1, current_k + 1):
        iterated_map.load_state_dict(torch.load(
            f'map_{idx_k}.pth'))
        for idx_batch in range(batch_num):
            if idx_k > 1:
                P_iterate = P_total[
                    batch_size * idx_batch:batch_size *
                    (idx_batch + 1)].to(device)
            else:
                P_iterate = p0_sampler(batch_size)

            P_total[batch_size * idx_batch:batch_size *
                    (idx_batch + 1)] = map_forward(iterated_map, P_iterate, sampling=True).detach()
        P_total = P_total.detach()
    return P_total


def new_pk_image_generator(iterated_map, current_k, num, dims, device):
    # To generate data with size [num, channel, img_size, img_size]
    batch_num = 60
    batch_size = int(num / batch_num)
    P_total = torch.randn([num, *dims])
    for idx_k in range(1, current_k + 1):
        iterated_map.load_state_dict(torch.load(
            f'map_{idx_k}.pth'))
        for idx_batch in range(batch_num):
            if idx_k > 1:
                P_iterate = P_total[
                    batch_size * idx_batch:batch_size *
                    (idx_batch + 1)].to(device)
            else:
                P_iterate = torch.randn([batch_size, *dims]).to(device)
            with torch.no_grad():
                P_total[batch_size * idx_batch:batch_size *
                        (idx_batch + 1)] = iterated_map(P_iterate).detach()
        P_total = P_total.detach()
    return P_total
