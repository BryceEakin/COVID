import torch as T
from .data import encode_protein, BatchMolGraph, create_protein_batch
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import gzip

import random
import itertools

from .utils import CHEMPROP_ARGS
import logging

__ALL__ = [
    'collate_stitch_data', 
    'StitchDataset',
    'create_dataloader',
    'create_data_split'
]

def collate_stitch_data(list_of_samples):
    names, chem_graphs, chem_features, proteins, results = zip(*list_of_samples)
    
    chem_graphs = BatchMolGraph(chem_graphs)
    chem_features = T.stack(chem_features)
    proteins = create_protein_batch(proteins)
    results = T.stack(results)
    
    return names, chem_graphs, chem_features, proteins, results
    
class StitchDataset(T.utils.data.Dataset):
    def __init__(self, base_folder):
        self.all_data = None
        self.all_proteins = None
        self.all_chemicals = None
        self.neg_samples = None
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
        chem_features = T.tensor(chem_features, dtype=T.float32)
        protein = encode_protein(self.all_proteins[row['item_id_b']])
        
        targets = T.tensor(row[['activation', 'binding', 'catalysis', 'inhibition', 'reaction']].values.astype(float), dtype=T.float32)
        
        return (row['item_id_a'], row['item_id_b']), chem_graph, chem_features, protein, targets


class SyntheticNegativeDataset(T.utils.data.Dataset):
    def __init__(self, pos_dataset, neg_rate=1.0):
        self.dataset = pos_dataset
        self.neg_rate = neg_rate
        self.pos_samples = None
        self.chem_options = None
        self.prot_options = None

    def __len__(self):
        return int(len(self.dataset) * self.neg_rate)

    def __getitem__(self, idx):
        if self.pos_samples is None:
            _ = len(self.dataset)
            self.pos_samples = set(
                self.dataset.all_data[['item_id_a', 'item_id_b']].itertuples(name=None, index=False)
            )
            self.chem_options = list(self.dataset.all_chemicals.keys())
            self.prot_options = list(self.dataset.all_proteins.keys())

        while True:
            chem = random.choice(self.chem_options)
            prot = random.choice(self.prot_options)

            if (chem, prot) not in self.pos_samples:
                break
        
        row = pd.Series([chem, prot] + [0.0]*5, index=self.dataset.all_data.columns)

        _,_,_,_, chem_graph, chem_features = self.dataset.all_chemicals[row['item_id_a']]
        chem_features = T.tensor(chem_features, dtype=T.float32)
        protein = encode_protein(self.dataset.all_proteins[row['item_id_b']])
        
        targets = T.tensor(row[['activation', 'binding', 'catalysis', 'inhibition', 'reaction']].values.astype(float), dtype=T.float32)
        
        return (chem, prot), chem_graph, chem_features, protein, targets


def create_dataloader(data, batch_size, sample_size=None, neg_rate=0.2, **dl_kwargs):
    class SubSampler(T.utils.data.Sampler):
        def __init__(self, length, numsamples):
            self._length = length
            self._numsamples = min(numsamples, length)

        def __iter__(self):
            return iter(np.random.choice(self._length, self._numsamples, replace=False))

        def __len__(self):
            return self._numsamples

    if neg_rate > 0.0:
        data = T.utils.data.ConcatDataset([data, SyntheticNegativeDataset(data, neg_rate)])

    return T.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        sampler=None if sample_size is None else SubSampler(len(data), sample_size),
        shuffle=True if sample_size is None else None,
        collate_fn = collate_stitch_data,
        **dl_kwargs
    )


def create_data_split(src_folder, 
                      dst_train_folder, 
                      dst_test_folder, 
                      pct_train=0.85, 
                      tolerance = 0.025, 
                      max_to_drop = 0.2, 
                      max_per_item=0.0025):
    data = StitchDataset(src_folder)
    data._deferred_load()
    logging.debug(f"Creating {pct_train} data split - {dst_train_folder} ; {dst_test_folder}")

    while True:
        all_data = data.all_data.copy()
        max_num_per_item = int(max_per_item * all_data.shape[0])

        # Make sure no element in the data is overwhelmingly present
        # Limit by chemical first
        all_data = all_data.groupby('item_id_a').apply(
            lambda x: x.sample(min(x.shape[0], max_num_per_item), replace=False)
        ).reset_index(drop=True)

        # Then by protein
        all_data = all_data.groupby('item_id_b').apply(
            lambda x: x.sample(min(x.shape[0], max_num_per_item), replace=False)
        ).reset_index(drop=True)

        logging.debug(f"{all_data.shape[0]}/{data.all_data.shape[0]} items remaining after max concentration filter")

        is_selected = pd.Series(False, index=all_data.index)

        protein_to_select = list(data.all_proteins.keys())
        random.shuffle(protein_to_select)

        have_selected = set()
        num_selected = 0

        while num_selected/is_selected.shape[0] < (1-pct_train) - tolerance and len(protein_to_select) > 0:
            prot = protein_to_select.pop(0)
            items_to_select = [prot]
            while items_to_select and num_selected/is_selected.shape[0] <= (1-pct_train) - tolerance:
                item = items_to_select.pop(0)
                print(f"#{len(have_selected)} ({num_selected/is_selected.shape[0]:0.2%}) - {item}                ", end='\r')
                
                have_selected.add(item)
                new_selections = (
                    (all_data['item_id_b'] == item)
                    | (all_data['item_id_a'] == item)
                )
                new_data = all_data.loc[new_selections]
                will_add = set(new_data['item_id_b'].values).union(
                    set(new_data['item_id_a'].values)
                ).difference(have_selected)

                if (is_selected | new_selections).mean() > (1-pct_train) + tolerance:
                    continue

                items_to_select.extend(
                    set(new_data['item_id_b'].values).union(
                        set(new_data['item_id_a'].values)
                    ).difference(have_selected)
                )
                is_selected = is_selected | new_selections
                num_selected = is_selected.sum()

        test_data = all_data.loc[is_selected]

        to_exclude = (
            (all_data['item_id_a'].isin(test_data['item_id_a'].unique()))
            | (all_data['item_id_b'].isin(test_data['item_id_b'].unique()))
        )
        
        if (to_exclude & ~is_selected).mean() > max_to_drop:
            logging.debug(f"Percent to drop outside of tolerance ({(to_exclude & ~is_selected).mean()}) -- retrying...")
            continue

        if is_selected.mean() <= (1-pct_train) + tolerance:
            break
        
        logging.debug(f"Test set outside of tolerance -- {is_selected.mean():0.3%} -- retrying...")

    logging.debug(f"Test set selected -- {is_selected.mean():0.3%} (dropping {(to_exclude & ~is_selected).mean():0.3%})")
    
    test_chem = {k:data.all_chemicals[k] for k in test_data['item_id_a'].unique()}
    test_prot = {k:data.all_proteins[k] for k in test_data['item_id_b'].unique()}

    train_data = all_data.loc[~(is_selected | to_exclude)]
    train_chem = {k:data.all_chemicals[k] for k in train_data['item_id_a'].unique()}
    train_prot = {k:data.all_proteins[k] for k in train_data['item_id_b'].unique()}
    
    assert len(set(train_prot.keys()).intersection(test_prot.keys())) == 0
    assert len(set(train_chem.keys()).intersection(test_chem.keys())) == 0
    
    if not os.path.exists(dst_train_folder):
        os.mkdir(dst_train_folder)
        
    train_data.to_csv(os.path.join(dst_train_folder, 'stitch_preprocessed.csv.gz'), index=False)
    with gzip.open(os.path.join(dst_train_folder, 'stitch_proteins.pkl.gz'), 'wb') as f:
        pkl.dump(train_prot, f, pkl.HIGHEST_PROTOCOL)
    with gzip.open(os.path.join(dst_train_folder, 'stitch_chemicals.pkl.gz'), 'wb') as f:
        pkl.dump(train_chem, f, pkl.HIGHEST_PROTOCOL)
        
    if not os.path.exists(dst_test_folder):
        os.mkdir(dst_test_folder)
        
    test_data.to_csv(os.path.join(dst_test_folder, 'stitch_preprocessed.csv.gz'), index=False)
    with gzip.open(os.path.join(dst_test_folder, 'stitch_proteins.pkl.gz'), 'wb') as f:
        pkl.dump(test_prot, f, pkl.HIGHEST_PROTOCOL)
    with gzip.open(os.path.join(dst_test_folder, 'stitch_chemicals.pkl.gz'), 'wb') as f:
        pkl.dump(test_chem, f, pkl.HIGHEST_PROTOCOL)
