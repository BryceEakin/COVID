import torch as T

def silu(x):
    return x * T.sigmoid(x)