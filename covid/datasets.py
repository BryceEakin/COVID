import torch as T
from .data import encode_protein, BatchMolGraph, create_protein_batch
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import gzip

from .utils import CHEMPROP_ARGS

__ALL__ = [
    'collate_stitch_data', 
    'StitchDataset'
]

def collate_stitch_data(list_of_samples):
    chem_graphs, chem_features, proteins, results = zip(*list_of_samples)
    
    chem_graphs = BatchMolGraph(chem_graphs, CHEMPROP_ARGS)
    chem_features = T.stack(chem_features)
    proteins = create_protein_batch(proteins)
    results = T.stack(results)
    
    return chem_graphs, chem_features, proteins, results
    
class StitchDataset(T.utils.data.Dataset):
    def __init__(self, base_folder="./data"):
        self.all_data = None
        self.all_proteins = None
        self.all_chemicals = None
        self._folder = base_folder
    
    def _deferred_load(self):
        self.all_data = pd.read_csv(
            os.path.join(self._folder, 'stitch_preprocessed.csv.gz'),
            dtype={
                'item_id_a': str,
                'item_id_b': str,
                'activation': np.float,
                'binding': np.float,
                'catalysis': np.float,
                'inhibition': np.float,
                'reaction': np.float
            }
        ).fillna(0.0)
        
        with gzip.open(os.path.join(self._folder, 'stitch_proteins.pkl.gz'), 'rb') as f:
            self.all_proteins = pkl.load(f)
        with gzip.open(os.path.join(self._folder, 'stitch_chemicals.pkl.gz'), 'rb') as f:
            self.all_chemicals = pkl.load(f)
    
    def __len__(self):
        if self.all_data is None:
            self._deferred_load()
        return self.all_data.shape[0]
    
    def __getitem__(self, idx):
        if self.all_data is None:
            self._deferred_load()
        row = self.all_data.iloc[idx]
        _,_,_,_, chem_graph, chem_features = self.all_chemicals[row['item_id_a']]
        chem_features = T.tensor(chem_features)
        protein = encode_protein(self.all_proteins[row['item_id_b']])
        
        targets = T.tensor(row[['activation', 'binding', 'catalysis', 'inhibition', 'reaction']].values.astype(float))
        
        return chem_graph, chem_features, protein, targets