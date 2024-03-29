import torch
from torch import nn
import numpy as np
from typing import List, Tuple


def init(module, weight_init, bias_init, gain=1.):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

RNN_TO_MODEL = {'LSTM': nn.LSTMCell, 'GRU': nn.GRUCell, 'RNN': nn.RNNCell}
ACTIVATION = {'relu': nn.ReLU, 'tanh': nn.Tanh}

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int,  hidden_sizes: List[int],
                 activation: str='relu', use_rnn: str=None, bias=False,
                 device: torch.device=torch.device('cpu')):
        super(MLP, self).__init__()
        assert activation in ['relu', 'tanh'], "activation_function must be one of ['relu', 'tanh']"
        assert isinstance(hidden_sizes, list), "hidden_sizes must be a list containing positive integers"
        assert len(hidden_sizes) > 0 , "hidden_sizes cannot be an empty list"

        self.hidden_sizes = hidden_sizes
        self.bias = bias
        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        layers = []
        previous_dim = input_size
        for dim in hidden_sizes:
            layers.append(init_(nn.Linear(previous_dim, dim, bias=bias)))
            layers.append(ACTIVATION[activation]())
            previous_dim = dim

        self.rnn = None
        if use_rnn:
            self.rnn = RNN_TO_MODEL[use_rnn](previous_dim, previous_dim, bias=bias)
        self.rnn_type = use_rnn

        self.net = nn.ModuleList(layers)

        self.output_layer = init_(nn.Linear(previous_dim, output_size, bias=bias))

    def forward(self, x: torch.tensor,  hc: Tuple[torch.tensor, torch.tensor] = None ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        for layer in self.net:
            x = layer(x)
        if self.rnn is not None:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            if self.rnn_type == 'LSTM':
                hc = self.rnn(x, hc)
            else:
                if hc is None:
                    hc = self.rnn(x, hc)
                else:
                    hc = self.rnn(x, hc[0])
                hc = (hc, hc)
            x = hc[0]
        x = self.output_layer(x)
        x = torch.tanh(x)
        return x, hc

