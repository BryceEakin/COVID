import torch as T
from torch import nn

from .utility import NONLINEARITIES, ResidualBlock, SiLU
from ..data import apply_to_protein_batch

__ALL__ = ['DownscaleConv1d', 'create_resnet_block_1d']

class DownscaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=2, nonlinearity='tanh', kernel=None, maxpool=False, for_protein_batch=False):
        super().__init__()
        
        if kernel is None:
            kernel = (downscale * 2 + 1, )
        elif not isinstance(kernel, tuple):
            kernel = (kernel, )

        wrap = (lambda x: x) if not for_protein_batch else apply_to_protein_batch
            
        self.conv = wrap(nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride=downscale,
            padding=tuple((x-1)//2 for x in kernel)
        ))
        
        if maxpool:
            self.pool = wrap(nn.MaxPool1d(downscale, ceil_mode=True, return_indices=False))
        else:
            self.pool = wrap(nn.AvgPool1d(downscale, ceil_mode=True, count_include_pad=False))
            
        self.kernel_size = kernel
        self.stride = downscale
        self.for_protein_batch = for_protein_batch
        
        self._nonlinearity = wrap(NONLINEARITIES[nonlinearity]())

    def forward(self, x):
        out = self.conv(x)

        if self.for_protein_batch:
            in_ch, out_ch = x.data.shape[1], out.data.shape[1]
            ch_match = min(in_ch, out_ch)
            
            pooled = self.pool(x[:,:ch_match,:])
            
            out[:,:ch_match,:] += pooled
        else:
            in_ch, out_ch = x.data.shape[1], out.data.shape[1]
            ch_match = min(in_ch, out_ch)

            out[:,:ch_match,:] += self.pool(x[:,:ch_match,:])
            
        return self._nonlinearity(out)

def create_resnet_block_1d(in_channels, inner_channels, inner_kernel=3, for_protein_batch = False, nonlinearity='silu'):
    wrap = (lambda x: x) if not for_protein_batch else apply_to_protein_batch
    
    if not isinstance(inner_kernel, tuple):
        inner_kernel = (inner_kernel, )
        
    inner = wrap(nn.Sequential(
        nn.Conv1d(in_channels, inner_channels, 1),
        nn.Conv1d(inner_channels,inner_channels, inner_kernel, padding=tuple((x-1)//2 for x in inner_kernel)),
        nn.Conv1d(inner_channels, in_channels, 1)
    ))

    for p in inner.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_normal_(p.data)
        else:
            nn.init.normal_(p.data)

    return ResidualBlock(inner, nonlinearity = wrap(NONLINEARITIES[nonlinearity]()))