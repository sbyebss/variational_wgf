import torch.nn as nn
import torch

####################### Basic accessories setup ###############################


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU(True)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softsign':  # map to [-1,1]
        return nn.Softsign()
    elif activation == 'Prelu':
        return nn.PReLU()
    elif activation == 'Rrelu':
        return nn.RReLU(0.5, 0.8)
    elif activation == 'hardshrink':
        return nn.Hardshrink()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanhshrink':
        return nn.Tanhshrink()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)

# * MLP

#! target for h/T


class Fully_connected(nn.Module):
    def __init__(self, input_dim=785, output_dim=1, hidden_dim=1024, num_layer=1, activation='Prelu', final_actv='Prelu', full_activ=True, bias=True, dropout=False, batch_nml=False, res=0, quadr=0, sigmoid=0):
        super(Fully_connected, self).__init__()
        self.full_activ = full_activ
        self.dropout = dropout
        self.batch_nml = batch_nml
        self.res = res
        assert quadr == 0 or sigmoid == 0
        self.quadr = quadr
        self.sigmoid = sigmoid

        self.layer1 = nn.Linear(
            input_dim, hidden_dim, bias=bias)
        self.layer1_activ = get_activation(activation)
        self.linearblock = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_layer)])
        self.atvt_list = nn.ModuleList(
            [get_activation(activation) for _ in range(num_layer)])
        if batch_nml:
            self.batchnormal = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(num_layer)])
        if dropout > 0:
            self.dropout_list = nn.ModuleList(
                [nn.Dropout(dropout) for _ in range(num_layer)])

        self.last_layer = nn.Linear(
            hidden_dim, output_dim, bias=bias)
        if self.full_activ:
            self.last_layer_activ = get_activation(final_actv)

    def forward(self, input):

        x = self.layer1_activ(self.layer1(input))

        for i in range(len(self.linearblock)):
            x = self.linearblock[i](x)
            if self.batch_nml:
                x = self.batchnormal[i](x)
            if self.dropout > 0:
                x = self.dropout_list[i](x)
            x = self.atvt_list[i](x)
        if self.full_activ:
            x = self.last_layer_activ(self.last_layer(x))
        else:
            x = self.last_layer(x)

        if self.res:
            x = x + input
        if self.quadr:
            return x**2
        elif self.sigmoid:
            return torch.sigmoid(3. * x / 20.) * 20.
        return x


class FC_linear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, res=0):
        super(FC_linear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.res = res
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim)

        self.linearblock = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        self.last_normal = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):

        x = self.fc1_normal(input)

        for i in range(self.num_layer):
            x = self.linearblock[i](x)

        x = self.last_normal(x)
        if self.res:
            return x + input
        return x
