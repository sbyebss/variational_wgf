import torch
import torch.nn as nn
from src.networks.dense_icnn import GradNN
from src.networks.mlp import get_activation


class ConvexLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(ConvexLinear, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):
        out = nn.functional.linear(input, self.weight, self.bias)
        return out


class ICNN(GradNN):
    def __init__(self, input_dim, hidden_dim, activation, num_layer, dropout=0, batch_size=1024):
        super(ICNN, self).__init__()
        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_layer = num_layer
        self.dropout = dropout

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        # begin to define my own normal and convex and activation
        self.normal = nn.ModuleList([nn.Linear(
            self.input_dim, self.hidden_dim, bias=True) for _ in range(2, self.num_layer + 1)])

        self.convex = nn.ModuleList([ConvexLinear(
            self.hidden_dim, self.hidden_dim, bias=False) for _ in range(2, self.num_layer + 1)])
        if dropout > 0:
            self.dropout_list = nn.ModuleList(
                [nn.Dropout(dropout) for _ in range(self.num_layer)])

        self.acti_list = nn.ModuleList(
            [get_activation(self.activation) for _ in range(2, self.num_layer + 1)])

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.store_positive_param()

    def store_positive_param(self):
        self.positive_params = []
        for p in list(self.parameters()):
            if hasattr(p, 'be_positive'):
                self.positive_params.append(p)

    def forward(self, input):
        x = self.activ_1(self.fc1_normal(input)).pow(2)

        for i in range(self.num_layer - 1):
            x = self.acti_list[i](self.convex[i](
                x).add(self.normal[i](input)))
            if self.dropout > 0:
                x = self.dropout_list[i](x)

        x = self.last_convex(x).add((0.5 * torch.norm(input, dim=1)**2).reshape(-1, 1))

        return x

    def convexify(self):
        for p in self.positive_params:
            p.data.clamp_(0)
