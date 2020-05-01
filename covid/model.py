import functools

import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn

from .data import ProteinBatchToPaddedBatch, apply_to_protein_batch
from .modules import (NONLINEARITIES, create_chemical_models,
                      create_protein_models)

import logging

__all__ = [
    "CovidModel",
    "RandomModel",
    'run_model',
]

class CovidModel(nn.Module):
    def __init__(self,
                 dropout:float = 0.2,
                 chem_nonlinearity:str = 'ReLU',
                 chem_hidden_size:int = 300,
                 chem_bias:bool = True,
                 chem_layers_per_message:int = 1,
                 chem_undirected:bool = False,
                 chem_atom_messages:bool = False,
                 chem_messages_per_pass:int = 2,
                 chem_mol_feature_dim:int = 211,
                 protein_base_dim:int = 32,
                 protein_output_dim:int = 256,
                 protein_nonlinearity:str = 'silu',
                 protein_downscale_nonlinearity:str = 'tanh',
                 protein_maxpool:bool = True,
                 negotiation_passes:int = 3,
                 context_dim:int = 256,
                 output_dim:int = 5
                 ):
        super().__init__()
        
        self.chem_head_model, self.chem_middle_model, self.chem_tail_model = create_chemical_models(
            activation=chem_nonlinearity,
            hidden_size=chem_hidden_size,
            context_size=context_dim,
            bias=chem_bias,
            dropout=dropout,
            layers_per_message=chem_layers_per_message,
            undirected=chem_undirected,
            atom_messages=chem_atom_messages,
            messages_per_pass=chem_messages_per_pass
        )
        
        raw_context_size = chem_mol_feature_dim + chem_hidden_size + protein_output_dim

        self.context_model = nn.Sequential(
            nn.Linear(raw_context_size, raw_context_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(raw_context_size, context_dim),
            nn.Tanh()
        )

        (
            self.protein_head_model,
            self.protein_middle_model,
            self.protein_tail_model
        ) = create_protein_models(protein_base_dim,
                                  context_dim,
                                  protein_output_dim,
                                  dropout,
                                  protein_nonlinearity,
                                  protein_downscale_nonlinearity,
                                  protein_maxpool)

        
        self.final_layers = nn.Sequential(
            nn.Linear(protein_output_dim + chem_hidden_size + chem_mol_feature_dim, context_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, output_dim),
            nn.Sigmoid()
        )

        self.num_passes = negotiation_passes
        
    def forward(self, chem_batch, chem_f_batch, protein_batch):
        chem_state = self.chem_head_model(chem_batch)
        protein_state = self.protein_head_model(protein_batch)

        if (~T.isfinite(chem_f_batch)).any():
            logging.debug("Encountered non-finite values in chem feature batch")
            with T.no_grad():
                chem_f_batch[~T.isfinite(chem_f_batch)] = 0.0

        for i in range(self.num_passes + 1):
            protein_context = self.protein_tail_model(protein_state)
            chem_context = self.chem_tail_model(chem_state)

            if (~T.isfinite(protein_context)).any():
                logging.debug("Protein context invalid!")
                with T.no_grad():
                    protein_context[~T.isfinite(protein_context)] = 0.0

            if (~T.isfinite(chem_context)).any():
                logging.debug("Chemical context invalid!")
                with T.no_grad():
                    chem_context[~T.isfinite(chem_context)] = 0.0

            context = T.cat((
                chem_context,
                chem_f_batch,
                protein_context
            ), -1)

            if i == self.num_passes:
                break

            context = self.context_model(context)

            chem_state = self.chem_middle_model(chem_state, context)
            protein_state = self.protein_middle_model(protein_state, context)
        

        return self.final_layers(context)


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
