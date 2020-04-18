import torch as T
from torch import nn
import torch.nn.functional as F

from .data import apply_to_protein_batch, ProteinBatchToPaddedBatch
from .modules import create_resnet_block_1d, DownscaleConv1d, Squeeze

import numpy as np
import functools

__all__ = [
    "CovidModel",
    "RandomModel",
    'create_protein_model',
    'run_model',
]

class CovidModel(nn.Module):
    def __init__(self, chem_model, protein_model, layers=2, dropout=0.0, in_dim=900, hidden_dim=512, out_dim=5):
        super().__init__()
        
        self.chem_model = chem_model
        self.protein_model = protein_model
        
        self.final_layers = nn.Sequential(*(
            [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ] + sum([[
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
            ] for _ in range(layers-1)],[])
            + [
                nn.Dropout(dropout),
                nn.Linear(hidden_dim,out_dim),
                nn.Sigmoid()
            ]
        ))
        
    def forward(self, chem_batch, chem_f_batch, protein_batch):
        chem_out = self.chem_model(chem_batch, chem_f_batch)
        protein_out = self.protein_model(protein_batch)
        
        result = self.final_layers(T.cat([chem_out, protein_out], -1))
        return result


class RandomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5,100),
            nn.ReLU(),
            nn.Linear(100,5),
            nn.Sigmoid()
        )
        
    def forward(self, chem_graphs, chem_features, proteins):
        batch_size = chem_graphs.n_mols
        input = T.rand((batch_size, 5), device=chem_features.device)
        return self.model(input)


def run_model(model, batch, device):
    _, chem_graphs, chem_features, proteins, target = batch
    
    chem_graphs = chem_graphs.to(device)
    chem_features = chem_features.to(device)
    proteins = proteins.to(device)

    weights = (target * 0.0015).clamp(1e-3, 1.0)
    weights[target == 0] = 0.5
    weights = weights.to(device)
    
    target = (1.0*(target > 0)).to(device)
    
    result = model(chem_graphs, chem_features, proteins)

    loss = F.binary_cross_entropy(result, target, weight=weights)
    return result, target, loss, weights


def create_protein_model(dropout = 0.2, 
                         outdim=600, 
                         base_dim=64,
                         nonlinearity='silu',
                         downscale_nonlinearity='tanh',
                         maxpool=True
                        ):

    resnet = functools.partial(create_resnet_block_1d, for_protein_batch=True, nonlinearity=nonlinearity)

    return nn.Sequential(
        # 21->100 channels inplace convolution
        apply_to_protein_batch(nn.Conv1d(23, base_dim * 4, (1, ), 1, 0)),
        apply_to_protein_batch(nn.Dropout(dropout)),
        
        #Do some resnet
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
        
        #Do some resnet
        resnet(base_dim * 8, base_dim * 2, inner_kernel=3),
        resnet(base_dim * 8, base_dim * 2, inner_kernel=5),
        
        # Scale it down again
        DownscaleConv1d(base_dim * 8, 
                        base_dim * 16,
                        4,
                        maxpool=True, 
                        for_protein_batch=True,
                        nonlinearity=downscale_nonlinearity),
        
        # Final resneting
        resnet(base_dim * 16, base_dim * 4, inner_kernel=7),
        resnet(base_dim * 16, base_dim * 4, inner_kernel=3),
        
        apply_to_protein_batch(nn.MaxPool1d(100000, ceil_mode=True)),
        
        # Convert protein batch to standard batch format
        ProteinBatchToPaddedBatch(),
        
        Squeeze(-1),
        nn.Dropout(dropout),
        nn.Linear(base_dim * 16, base_dim * 16),
        nn.ReLU(),
        nn.Linear(base_dim * 16, outdim),
        #nn.Tanhshrink(),
        #nn.Tanh(),
    )