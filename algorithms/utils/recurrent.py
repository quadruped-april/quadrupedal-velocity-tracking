import copy

from torch import nn

from .module import MLP

__all__ = ['is_recurrent', 'RMlp']


def is_recurrent(*modules):
    return any(
        getattr(m, 'is_recurrent', False) or
        isinstance(m, nn.RNNBase)
        for m in modules
    )


class RMlp(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        rnn_type,
        rnn_num_layers,
        rnn_hidden_dim,
        mlp_shape,
        activation_fn,
        input_size,
        output_size,
        final_activation=False,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                num_layers=rnn_num_layers,
                hidden_size=rnn_hidden_dim,
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                num_layers=rnn_num_layers,
                hidden_size=rnn_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown RNN Type {rnn_type}")

        self.mlp = MLP(mlp_shape, activation_fn, rnn_hidden_dim, output_size, final_activation)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, x, h=None):
        # for x.dim() == 2, treat the 1st dim as batch instead of time
        if x.dim() < 3:
            x = x.unsqueeze(0)
        f, hn = self.rnn(x, h)
        return self.mlp(f.squeeze(0)), hn

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        result.rnn.flatten_parameters()
        return result
