from torch import nn
from ..functions import silu

__ALL__ = ['SiLU', 'NONLINEARITIES', 'Squeeze', 'ResidualBlock']

class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)

NONLINEARITIES = {'relu': nn.ReLU, 'silu': SiLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

class Squeeze(nn.Module):
    def __init__(self, axis):
        self.axis=axis
        
    def __call__(self, x):
        return x.squeeze(self.axis)

class ResidualBlock(nn.Module):
    def __init__(self, block, nonlinearity='silu'):
        super().__init__()
        self.model_layer = block
        if isinstance(nonlinearity, nn.Module):
            self._nonlinearity = nonlinearity
        else:
            self._nonlinearity = NONLINEARITIES[nonlinearity]()
        
    def forward(self, x):
        return self._nonlinearity(x + self.model_layer(x))