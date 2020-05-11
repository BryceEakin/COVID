import functools

import torch as T
from torch import nn

from ..data import ProteinBatchToPaddedBatch, apply_to_protein_batch, ProteinBatch
from .conv import DownscaleConv1d, create_resnet_block_1d
from .utility import NONLINEARITIES, ResidualBlock, Squeeze

__ALL__ = [
    'ProteinHeadModel',
    'create_protein_middle_model',
    'create_protein_tail_model',
    'create_protein_models'
]

class ProteinMultiheadAttention(nn.Module):
    def __init__(self, num_channels, num_heads, window):
        super().__init__()

        self.conv_heads = apply_to_protein_batch(
            nn.Conv1d(num_channels, num_heads, (window, ), 1, (window-1)//2)
        )

    def forward(self, x):
        return self.conv_heads(x).batchwise_apply(T.softmax, -1)

class ProteinMHAttentionTransformer(ProteinMultiheadAttention):
    def __init__(self, num_channels, num_heads, window):
        super().__init__(num_channels, num_heads, window)

        self.result_transform = nn.Conv1d(
            num_channels * num_heads, num_channels * num_heads, (1, ), 1, 0, groups=num_heads
        )

    def forward(self, x):
        focuses = super().forward(x)

        # Create B x (Heads x Channels) x L attention-data tensor
        focused_data = T.flatten(focuses._data.unsqueeze(2) * x._data.unsqueeze(1), 1, 2)

        # Transform the tensor and sum along the 'heads' dimention
        transform_result = self.result_transform(focused_data.contiguous()).view(
            1, focuses.num_channels, x.num_channels, x._data.shape[-1]
        ).sum(1).contiguous()

        return ProteinBatch(transform_result, x.batch_offsets, x.batch_lengths)

class ProteinMHAttentionSummarizer(ProteinMultiheadAttention):
    def __init__(self, num_channels, num_heads, window, reduce='max'):
        super().__init__(num_channels, num_heads, window)

        if reduce not in ('max', 'mean', 'sum'):
            raise ValueError('Unknown reduction function -- expected max, mean, or sum')
        
        self.reduce = reduce

    def forward(self, x):
        reduce = getattr(T, self.reduce)
        if self.reduce == 'max':
            def reduce(*args, **kwargs):
                return T.max(*args, **kwargs)[0]
            
        focuses = super().forward(x)

        return reduce(T.cat([
            (focuses[
                (slice(None), slice(head_idx, head_idx + 1), slice(None))
            ] * x).batchwise_apply(T.sum, -1, keepdim=True, output_protein=False)
            for head_idx in range(focuses.num_channels)
        ], -1), -1)


class ProteinHeadModel(nn.Module):
    def __init__(self,
                 base_dim=64,
                 dropout=0.2,
                 nonlinearity='silu',
                 downscale_nonlinearity='tanh',
                 maxpool=True):
        super().__init__()

        resnet = functools.partial(
            create_resnet_block_1d, 
            for_protein_batch=True, 
            nonlinearity=nonlinearity,
            norm='instance'
        )

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
                 nonlinearity='silu',
                 num_attention_layers=3,
                 num_attention_heads=8,
                 attention_window=3):
        super().__init__()

        resnet = functools.partial(
            create_resnet_block_1d, 
            for_protein_batch=True, 
            nonlinearity=nonlinearity,
            norm='batch'
        )

        self.context_model = nn.Sequential(
            nn.BatchNorm1d(context_dim),
            nn.Dropout(dropout),
            nn.Linear(context_dim, input_dim),
            NONLINEARITIES[nonlinearity](),
            nn.Linear(input_dim, input_dim)
        )

        self.model = nn.Sequential(
            *[
                apply_to_protein_batch(nn.Dropout(dropout)),

                # inplace convolution
                apply_to_protein_batch(nn.Conv1d(input_dim, input_dim, (1, ), 1, 0)),

            ] + [

                # For each attention layer...
                ResidualBlock(nn.Sequential(
                    apply_to_protein_batch(nn.BatchNorm1d(input_dim)),
                    apply_to_protein_batch(NONLINEARITIES[nonlinearity]()),
                    ProteinMHAttentionTransformer(input_dim, num_attention_heads, attention_window),
                    resnet(input_dim, input_dim // 4, inner_kernel=3),
                    resnet(input_dim, input_dim // 4, inner_kernel=5),
                ), None)
                for _ in range(num_attention_layers)
            ] + [
            
                # inplace convolution
                apply_to_protein_batch(nn.Conv1d(input_dim, input_dim, (1, ), 1, 0)),
            ]
        )

    def forward(self, state, context):
        channel_reweighting = T.tanh(self.context_model(context)) + 1.0 + 1e-8
        
        # Ensure, on average, signal doesn't change
        channel_reweighting = (
            channel_reweighting / channel_reweighting.mean(-1, keepdim=True)
        ).unsqueeze(-1)
        
        return self.model(state * channel_reweighting)


def create_protein_model_middle(input_dim=512,
                                context_dim=256,
                                dropout=0.2,
                                nonlinearity='silu',
                                attention_layers=3,
                                attention_heads=8,
                                attention_window=3):
    return ResidualBlock(_ProteinMiddleModel(
        input_dim, context_dim, dropout, nonlinearity, attention_layers, attention_heads, attention_window
    ), None)


def create_protein_model_tail(input_dim=512,
                              output_dim=256,
                              dropout=0.2,
                              nonlinearity='silu',
                              output_attention=True,
                              output_attention_heads=16,
                              output_attention_window=1):

    resnet = functools.partial(create_resnet_block_1d, for_protein_batch=True, nonlinearity=nonlinearity)

    if not output_attention:
        # Take the max on each channel
        summarizer = nn.Sequential(
            apply_to_protein_batch(nn.MaxPool1d(100000, ceil_mode=True)),
            # Convert protein batch to standard batch format
            ProteinBatchToPaddedBatch(),
            Squeeze(-1),
        )

    else:
        summarizer = ProteinMHAttentionSummarizer(input_dim, output_attention_heads, output_attention_window)

    return nn.Sequential(
        apply_to_protein_batch(nn.Dropout(dropout)),

        # Final resneting
        resnet(input_dim, input_dim // 4, inner_kernel=3),
        resnet(input_dim, input_dim // 4, inner_kernel=3),
        
        # Converts from protein batch to standard tensor here
        apply_to_protein_batch(nn.BatchNorm1d(input_dim)),
        apply_to_protein_batch(NONLINEARITIES[nonlinearity]()),

        summarizer,

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
                          maxpool=True,
                          attention_layers=3,
                          attention_heads=8,
                          attention_window=3,
                          output_use_attention=True,
                          output_attention_heads=16):
    head = ProteinHeadModel(base_dim, dropout, nonlinearity, downscale_nonlinearity, maxpool)
    middle = create_protein_model_middle(
        base_dim * 16, context_dim, dropout, nonlinearity, attention_layers, attention_heads, attention_window
    )
    tail = create_protein_model_tail(
        base_dim * 16, output_dim, dropout, nonlinearity, output_use_attention, output_attention_heads
    )

    return head, middle, tail



        
