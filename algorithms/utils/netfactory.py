from typing import Iterable, Literal

from torch import nn

from .module import MLP
from .recurrent import RMlp

__all__ = ['NetworkFactory', 'make_nn']


class NetworkFactory:
    def __init__(
        self,
        shape: Iterable,
        activation: str = 'ReLU',
        type: Literal['mlp', 'lstm', 'gru'] = 'mlp',
        rnn_hidden_dim: int = -1,
        rnn_num_layers: int = -1,
        stochastic=False,
        **kwargs,
    ):
        self.shape = tuple(shape)
        self.activation = activation
        self.type = type.lower()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.stochastic = stochastic
        self.kwargs = kwargs
        self.nn = None
        self.input_dim = None
        self.output_dim = None

    @classmethod
    def make(cls, cfg: dict, input_dim, output_dim):
        obj = cls(**cfg)
        obj(input_dim, output_dim)
        return obj

    def __call__(self, input_dim, output_dim):
        self.input_dim, self.output_dim = input_dim, output_dim
        activation_fn = getattr(nn, self.activation)
        if self.is_rnn() and self.rnn_hidden_dim < 0:
            self.rnn_hidden_dim = self.shape[0]
            self.shape = self.shape[1:]

        if self.type == 'mlp':  # mlp
            self.nn = MLP(
                self.shape, activation_fn, input_dim, output_dim,
            )
        else:
            if self.rnn_num_layers <= 0:
                raise ValueError(f'For {self.type}, {self.rnn_num_layers=} <= 0!')

            self.nn = RMlp(
                self.type,
                self.rnn_num_layers,
                self.rnn_hidden_dim,
                self.shape,
                activation_fn,
                input_dim,
                output_dim,
            )
        return self.nn

    def is_rnn(self):
        return self.type == 'lstm' or self.type == 'gru'


def make_nn(cfg: dict, input_dim, output_dim, **kwargs):
    return NetworkFactory.make(cfg | kwargs, input_dim, output_dim).nn
