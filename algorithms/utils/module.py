import torch
from torch import nn

__all__ = ['MLP', 'default_device']


class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_dim, output_dim, final_activation=False):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn

        modules = []
        shape = [input_dim] + list(shape)
        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            modules.append(self.activation_fn())
        modules.append(nn.Linear(shape[-1], output_dim))
        if final_activation:
            modules.append(self.activation_fn())
        self.architecture = nn.Sequential(*modules)

        self.shape = shape
        self.input_shape = [input_dim]
        self.output_shape = [output_dim]

    def forward(self, x):
        return self.architecture(x)


def default_device(device=None):
    if device == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
