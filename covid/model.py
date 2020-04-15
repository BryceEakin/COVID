import torch as T
from torch import nn
import torch.nn.functional as F

from .data import apply_to_protein_batch, ProteinBatchToPaddedBatch
from .modules import create_resnet_block_1d, DownscaleConv1d, Squeeze

import numpy as np

__all__ = [
    "CovidModel",
    "RandomModel",
    'create_protein_model',
    'run_model',
    'calculate_average_loss_and_accuracy'
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
    chem_graphs, chem_features, proteins, target = batch
    
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


def create_protein_model(dropout = 0.2, outdim=600):
    return nn.Sequential(
        # 21->100 channels inplace convolution
        apply_to_protein_batch(nn.Conv1d(23, 512, (1, ), 1, 0)),
        apply_to_protein_batch(nn.Dropout(dropout)),
        
        #Do some resnet
        create_resnet_block_1d(512, 64, inner_kernel=3, for_protein_batch=True),
        create_resnet_block_1d(512, 64, inner_kernel=5, for_protein_batch=True),
        apply_to_protein_batch(nn.Dropout(dropout)),

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

def calculate_average_loss_and_accuracy(model, dl, device):
    model.eval()
    
    total_loss = 0.0
    total_div = 0
    
    accuracy_acc = np.zeros(5)
    accuracy_div = np.zeros(5)
    
    tp = np.zeros(5)
    fp = np.zeros(5)
    fn = np.zeros(5)
    tn = np.zeros(5)
    
    for batch in dl:
        chem_graphs, chem_features, proteins, target = batch
        result, target, loss, weight = run_model(model, batch, device)
        
        total_loss += loss.item() * result.shape[0]
        total_div += result.shape[0]
        
        accuracy_acc += (((result > 0.5) == target) * weight).sum(0).cpu().numpy()
        accuracy_div += weight.sum(0).cpu().numpy()
        
        tp += (((result >= 0.5) * target) * weight).sum(0).cpu().numpy()
        fp += (((result >= 0.5) * (1 - target)) * weight).sum(0).cpu().numpy()
        fn += (((result < 0.5) * target) * weight).sum(0).cpu().numpy()
        tn += (((result < 0.5) * (1 - target)) * weight).sum(0).cpu().numpy()
        
    return total_loss / total_div, accuracy_acc / accuracy_div, (tp,fp,fn,tn)