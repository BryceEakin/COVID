import argparse

__ALL__ = ['CHEMPROP_ARGS']

def create_chemprop_args():
    args = argparse.Namespace()
    args.seed = 0
    args.ensemble_size = 1
    args.hidden_size = 300
    args.bias = False
    args.depth = 3
    args.dropout = 0.0
    args.activation = 'ReLU'
    args.undirected = False
    args.atom_messages = False
    
    return args

CHEMPROP_ARGS = create_chemprop_args()