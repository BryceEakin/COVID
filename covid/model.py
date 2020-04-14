import torch as T
from torch import nn

from .data import apply_to_protein_batch, ProteinBatchToPaddedBatch
from .modules import create_resnet_block_1d, DownscaleConv1d, Squeeze


class CovidModel(nn.Module):
    def __init__(self, chem_model, protein_model, in_dim=900, dropout=0.0, out_dim=5):
        super().__init__()
        
        self.chem_model = chem_model
        self.protein_model = protein_model
        
        self.final_layers = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, chem_batch, chem_f_batch, protein_batch):
        chem_out = self.chem_model(chem_batch, chem_f_batch)
        protein_out = self.protein_model(protein_batch)
        
        result = self.final_layers(T.cat([chem_out, protein_out], -1))
        return result


def create_protein_model(dropout = 0.2, outdim=600):
    return nn.Sequential(
        # 21->100 channels inplace convolution
        apply_to_protein_batch(nn.Conv1d(23, 512, (1, ), 1, 0)),
        
        #Do some resnet
        create_resnet_block_1d(512, 64, inner_kernel=3, for_protein_batch=True),
        create_resnet_block_1d(512, 64, inner_kernel=5, for_protein_batch=True),
        create_resnet_block_1d(512, 64, inner_kernel=7, for_protein_batch=True),
        create_resnet_block_1d(512, 64, inner_kernel=11, for_protein_batch=True),
        
        # Scale it down
        DownscaleConv1d(512, 512, 4, maxpool=True, for_protein_batch=True),
        apply_to_protein_batch(nn.Dropout(dropout)),
        
        #Do some resnet
        create_resnet_block_1d(512, 128, inner_kernel=3, for_protein_batch=True),
        create_resnet_block_1d(512, 128, inner_kernel=5, for_protein_batch=True),
        
        # Scale it down again
        DownscaleConv1d(512, 1024,4,'silu', maxpool=True, for_protein_batch=True),
        
        # Final resneting
        create_resnet_block_1d(1024, 256, inner_kernel=7, for_protein_batch=True),
        create_resnet_block_1d(1024, 256, inner_kernel=3, for_protein_batch=True),
        
        apply_to_protein_batch(nn.MaxPool1d(100000, ceil_mode=True)),
        
        # Convert protein batch to standard batch format
        ProteinBatchToPaddedBatch(),
        
        Squeeze(-1),
        nn.Dropout(dropout),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024,outdim),
        #nn.Tanhshrink(),
        #nn.Tanh(),
    )