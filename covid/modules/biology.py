import functools

import torch as T
from torch import nn

from ..data import ProteinBatchToPaddedBatch, apply_to_protein_batch
from .conv import DownscaleConv1d, create_resnet_block_1d
from .utility import NONLINEARITIES, ResidualBlock, Squeeze

__ALL__ = [
    'ProteinHeadModel',
    'create_protein_middle_model',
    'create_protein_tail_model',
    'create_protein_models'
]

class ProteinHeadModel(nn.Module):
    def __init__(self,
                 base_dim=64,
                 dropout=0.2,
                 nonlinearity='silu',
                 downscale_nonlinearity='tanh',
                 maxpool=True):
        super().__init__()

        resnet = functools.partial(create_resnet_block_1d, for_protein_batch=True, nonlinearity=nonlinearity)

        self.module = nn.Sequential(
            # 21->100 channels inplace convolution
            apply_to_protein_batch(nn.Conv1d(23, base_dim * 4, (1, ), 1, 0)),
            apply_to_protein_batch(nn.Dropout(dropout)),
            
            # Do some resnet
            resnet(base_dim * 4, base_dim, inner_kernel=3),
            resnet(base_dim * 4, base_dim, inner_kernel=5),
            apply_to_protein_batch(nn.Dropout(dropout)),

            resnet(base_dim * 4, base_dim, inner_kernel=7),
            resnet(base_dim * 4, base_dim, inner_kernel=11),
            
            # Scale it down
            DownscaleConv1d(base_dim * 4, 
                            base_dim * 8, 
                            4, 
                            maxpool=True, 
                            for_protein_batch=True, 
                            nonlinearity=downscale_nonlinearity),
            apply_to_protein_batch(nn.Dropout(dropout)),

            # Do some resnet
            resnet(base_dim * 8, base_dim * 2, inner_kernel=3),
            resnet(base_dim * 8, base_dim * 2, inner_kernel=5),
            
            # Scale it down again
            DownscaleConv1d(base_dim * 8, 
                            base_dim * 16,
                            4,
                            maxpool=True, 
                            for_protein_batch=True,
                            nonlinearity=downscale_nonlinearity),
            
            apply_to_protein_batch(nn.Dropout(dropout))
        )

    def forward(self, x):
        return self.module(x)

class _ProteinMiddleModel(nn.Module):
    def __init__(self, 
                 input_dim=512,
                 context_dim=256,
                 dropout=0.2,
                 nonlinearity='silu'):
        super().__init__()

        resnet = functools.partial(create_resnet_block_1d, for_protein_batch=True, nonlinearity=nonlinearity)

        self.context_model = nn.Sequential(
            nn.Linear(context_dim, input_dim),
            NONLINEARITIES[nonlinearity](),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2)
        )

        self.model = nn.Sequential(
            # inplace convolution
            apply_to_protein_batch(nn.Conv1d(input_dim, input_dim, (1, ), 1, 0)),

            #Do some resnet
            resnet(input_dim, input_dim // 4, inner_kernel=3),
            resnet(input_dim, input_dim // 4, inner_kernel=5),
            apply_to_protein_batch(nn.Dropout(dropout)),

            resnet(input_dim, input_dim // 4, inner_kernel=3),
            resnet(input_dim, input_dim // 4, inner_kernel=5),
            apply_to_protein_batch(nn.Dropout(dropout)),

            # inplace convolution
            apply_to_protein_batch(nn.Conv1d(input_dim, input_dim, (1, ), 1, 0)),
        )

    def forward(self, state, context):
        channel_reweighting = self.context_model(context)
        
        # Reweight half of the channels to avoid vanishing gradients
        channel_reweighting = T.cat(
            (channel_reweighting, T.ones_like(channel_reweighting)), 
            -1
        ).unsqueeze(-1)

        state = state * channel_reweighting

        return self.model(state)

def create_protein_model_middle(input_dim=512,
                                context_dim=256,
                                dropout=0.2,
                                nonlinearity='silu'):
    return ResidualBlock(
        _ProteinMiddleModel(input_dim, context_dim, dropout, nonlinearity),
        apply_to_protein_batch(NONLINEARITIES[nonlinearity]())
    )

def create_protein_model_tail(input_dim=512,
                              output_dim=256,
                              dropout=0.2,
                              nonlinearity='silu'):

    resnet = functools.partial(create_resnet_block_1d, for_protein_batch=True, nonlinearity=nonlinearity)

    return nn.Sequential(
        # Final resneting
        resnet(input_dim, input_dim // 4, inner_kernel=3),
        resnet(input_dim, input_dim // 4, inner_kernel=3),
        
        # Take the max on each channel
        apply_to_protein_batch(nn.MaxPool1d(100000, ceil_mode=True)),
        
        # Convert protein batch to standard batch format
        ProteinBatchToPaddedBatch(),
        
        Squeeze(-1),
        nn.Dropout(dropout),
        nn.Linear(input_dim, input_dim),
        NONLINEARITIES[nonlinearity](),
        nn.Linear(input_dim, output_dim),
    )

def create_protein_models(base_dim=16,
                          context_dim=256,
                          output_dim=256,
                          dropout=0.2,
                          nonlinearity='silu',
                          downscale_nonlinearity='tanh',
                          maxpool=True):
    head = ProteinHeadModel(base_dim, dropout, nonlinearity, downscale_nonlinearity, maxpool)
    middle = create_protein_model_middle(base_dim * 16, context_dim, dropout, nonlinearity)
    tail = create_protein_model_tail(base_dim * 16, output_dim, dropout, nonlinearity)

    return head, middle, tail