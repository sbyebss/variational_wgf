import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm


class ConvexQuadratic(nn.Module):
    '''Convex Quadratic Layer'''
    __constants__ = ['in_features', 'out_features', 'quadratic_decomposed', 'weight', 'bias']

    def __init__(self, in_features, out_features, bias=True, rank=1):
        super(ConvexQuadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.quadratic_decomposed = nn.Parameter(torch.Tensor(
            torch.randn(in_features, rank, out_features)
        ))
        self.weight = nn.Parameter(torch.Tensor(
            torch.randn(out_features, in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # self.linear = nn.Linear(in_features, out_features)

    def forward(self, input):
        quad = ((input.matmul(self.quadratic_decomposed.transpose(1, 0)).transpose(1, 0)) ** 2).sum(dim=1)
        linear = F.linear(input, self.weight, self.bias)
        # linear = self.linear(input)
        return quad + linear


class WeightTransformedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, w_transform=lambda x: x):
        super().__init__(in_features, out_features, bias=bias)
        self._w_transform = w_transform

    def forward(self, input):
        return F.linear(input, self._w_transform(self.weight), self.bias)


class GradNN(nn.Module):
    def __init__(self, batch_size=1024):
        super(GradNN, self).__init__()
        self.batch_size = batch_size

    def forward(self, input):
        pass

    def push(self, input, create_graph=True, retain_graph=True):
        '''
        Pushes input by using the gradient of the network. By default preserves the computational graph.
        # Apply to small batches.
        '''
        if len(input) <= self.batch_size:
            output = autograd.grad(
                outputs=self.forward(input), inputs=input,
                create_graph=create_graph, retain_graph=retain_graph,
                only_inputs=True,
                grad_outputs=torch.ones_like(input[:, :1], requires_grad=False)
            )[0]
            return output
        else:
            output = torch.zeros_like(input, requires_grad=False)
            for j in range(0, input.size(0), self.batch_size):
                output[j: j + self.batch_size] = self.push(
                    input[j:j + self.batch_size],
                    create_graph=create_graph, retain_graph=retain_graph)
            return output

    def push_nograd(self, input):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        for i in range(0, len(input), self.batch_size):
            output.data[i:i + self.batch_size] = self.push(
                input[i:i + self.batch_size],
                create_graph=False, retain_graph=False
            ).data
        return output

    def hessian(self, input):
        gradient = self.push(input)
        hessian = torch.zeros(
            *gradient.size(), self.dim,
            dtype=torch.float32,
            requires_grad=True,
        )

        hessian = torch.cat(
            [
                torch.autograd.grad(
                    outputs=gradient[:, d], inputs=input,
                    create_graph=True, retain_graph=True,
                    only_inputs=True, grad_outputs=torch.ones(input.size()[0]).float().to(input)
                )[0][:, None, :]
                for d in range(self.dim)
            ],
            dim=1
        )
        return hessian


class DenseICNN(GradNN):
    '''Fully Conncted ICNN with input-quadratic skip connections.'''

    def __init__(
        self, dim,
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu',
        strong_convexity=1e-6,
        batch_size=1024,
        conv_layers_w_trf=lambda x: x,
        forse_w_positive=True
    ):
        super(DenseICNN, self).__init__(batch_size)

        self.dim = dim
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.rank = rank
        self.conv_layers_w_trf = conv_layers_w_trf
        self.forse_w_positive = forse_w_positive

        self.quadratic_layers = nn.ModuleList([
            ConvexQuadratic(dim, out_features, rank=rank, bias=True)
            for out_features in hidden_layer_sizes
        ])

        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            WeightTransformedLinear(
                in_features, out_features, bias=False, w_transform=self.conv_layers_w_trf)
            for (in_features, out_features) in sizes
        ])

        self.final_layer = WeightTransformedLinear(
            hidden_layer_sizes[-1], 1, bias=False, w_transform=self.conv_layers_w_trf)

    def forward(self, input):
        '''Evaluation of the discriminator value. Preserves the computational graph.'''
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            elif self.activation == 'relu':
                output = F.relu(output)
            else:
                raise Exception('Activation is not specified or unknown.')

        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)

    def convexify(self):
        if self.forse_w_positive:
            for layer in self.convex_layers:
                if (isinstance(layer, nn.Linear)):
                    layer.weight.data.clamp_(0)
            self.final_layer.weight.data.clamp_(0)


from IPython.display import clear_output


def id_pretrain_model(
        model, sampler, lr=1e-3, n_max_iterations=2000, batch_size=1024, loss_stop=1e-5, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    for it in tqdm(range(n_max_iterations), disable=not verbose):
        X = sampler(batch_size)
        if len(X.shape) == 1:
            X = X.view(-1, 1)
        X.requires_grad_(True)
        loss = F.mse_loss(model.push(X), X)
        loss.backward()

        opt.step()
        opt.zero_grad()
        model.convexify()

        if verbose:
            if it % 100 == 99:
                clear_output(wait=True)
                print('Loss:', loss.item())

            if loss.item() < loss_stop:
                clear_output(wait=True)
                print('Final loss:', loss.item())
                break
    return model
