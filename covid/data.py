import torch as T
from torch import nn
import torch.nn.functional as F

import typing as typ
import numpy as np
import math

from torch.nn.modules import pooling 
from torch.nn.modules.conv import _ConvNd

__ALL__ = [
    'ProteinBatch', 
    'encode_protein', 
    'create_protein_batch',
    'apply_to_protein_batch',
    'ApplyConvToProteinBatch',
    'ApplyPoolToProteinBatch',
    'ApplyToProteinBatch',
    'ApplyReducingToProteinBatch'
    'protein_batch_to_padded_batch',
    'ProteinBatchToPaddedBatch'
]

POOL_ATTRIBUTES = {'kernel_size', 'stride', 'padding', 'ceil_mode'}

VALID_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWXY'
AMINO_ACID_INDICES = {aa:idx for idx,aa in enumerate(VALID_AMINO_ACIDS)}

def encode_protein(protein_str):
    x = T.zeros(23, len(protein_str), dtype=T.float, requires_grad=False)
    p_indices = list(map(AMINO_ACID_INDICES.get, protein_str))
    x[p_indices, T.arange(len(protein_str))] = 1.0
    return x


def create_protein_batch(sequences: typ.List[T.Tensor], padding: int=5):
    batch_lengths = [x.shape[-1] for x in sequences]
    batch_offsets = [0] + list(padding * (np.arange(len(sequences))+1) + np.cumsum(batch_lengths))[:-1]        
    data = T.cat([F.pad(x, (0, padding)) for x in sequences], -1)
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
        
    return ProteinBatch(data.contiguous(), batch_offsets, batch_lengths)


class ProteinBatch(object):
    def __init__(self, data: T.Tensor, batch_offsets=typ.List[int], batch_lengths=typ.List[int]):
        
        if len(batch_offsets) != len(batch_lengths):
            raise ValueError(f"offsets and lengths to ProteinBatch must be same length: {batch_offsets}; {batch_lengths}")
        
        if sum(batch_lengths) > data.shape[-1]:
            raise ValueError(f"Supplied lengths inconsistent with supplied tensor to ProteinBatch: {batch_offsets}; {batch_lengths}")
            
        if any(batch_offsets[i] + batch_lengths[i] > batch_offsets[i+1] for i in range(len(batch_offsets)-1)):
            raise ValueError(f"batch definitions to ProteinBatch are not sorted or are overlapping: {batch_offsets}; {batch_lengths}")
            
        if batch_lengths[-1] + batch_offsets[-1] > data.shape[-1]:
            raise ValueError(f"Supplied batch definitions out of bounds for suppied tensor to ProteinBatch: {batch_offsets}; {batch_lengths}")
        
        self._data = data
        self._batch_offsets = tuple(batch_offsets)
        self._batch_lengths = tuple(batch_lengths)
        
    @property
    def data(self):
        return self._data
    
    @property
    def batch_lengths(self):
        return self._batch_lengths
    
    @property
    def batch_offsets(self):
        return self._batch_offsets
        
    def broadcast_like(self, other):
        if not isinstance(other, ProteinBatch):
            if other.shape == self.data.shape:
                return self
            raise ValueError("Can't broadcast to " + repr(other))
            
        if (other.data.shape == self.data.shape
                and len(other.batch_lengths) == len(self.batch_lengths)
                and all(x == y for x, y in zip(other.batch_lengths, self.batch_lengths))
                and all(x == y for x, y in zip(other.batch_offsets, self.batch_offsets))):
            return self
            
        if any(x != y for x,y in zip(self._batch_lengths, other._batch_lengths)):
            raise ValueError("Can't broadcast different length sequences in ProteinBatch objects")
            
        new_data = T.zeros_like(other.data)
        for from_off, length, to_off in zip(self._batch_offsets, self._batch_lengths, other._batch_offsets):
            new_data[:,:,to_off:(to_off+length)] = self.data[:,:,from_off:(from_off+length)]
            
        return ProteinBatch(new_data, other._batch_offsets, other._batch_lengths)
        
    def __repr__(self):
        return f"<ProteinBatch[{len(self.batch_lengths)}]({self.data.shape}) offsets={self.batch_offsets}, lengths={self.batch_lengths}>"
        
    def __add__(self, other):
        if isinstance(other, ProteinBatch):
            other = other.broadcast_like(self)
            return ProteinBatch(self.data + other.data, self.batch_offsets, self.batch_lengths)
        
        return ProteinBatch(self.data + other, self.batch_offsets, self.batch_lengths)
    
    def __sub__(self, other):
        if isinstance(other, ProteinBatch):
            other = other.broadcast_like(self)
            return ProteinBatch(self.data - other.data, self.batch_offsets, self.batch_lengths)
        
        return ProteinBatch(self.data - other, self.batch_offsets, self.batch_lengths)
    
    def __mul__(self, other):
        if isinstance(other, ProteinBatch):
            other = other.broadcast_like(self)
            return ProteinBatch(self.data * other.data, self.batch_offsets, self.batch_lengths)
        
        return ProteinBatch(self.data * other, self.batch_offsets, self.batch_lengths)
    
    def __truediv__(self, other):
        if isinstance(other, ProteinBatch):
            other = other.broadcast_like(self)
            return ProteinBatch(self.data / other.data, self.batch_offsets, self.batch_lengths)
        
        return ProteinBatch(self.data / other, self.batch_offsets, self.batch_lengths)
    
    def __neg__(self, other):
        return ProteinBatch(-self.data, self.batch_offsets, self.batch_lengths)
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise ValueError("Cannot slice on the batch index of a ProteinBatch")
        elif isinstance(idx, tuple):
            if idx[0] != slice(None):
                raise ValueError("Cannot slice on the batch index of a ProteinBatch")
            if len(idx) > 3:
                raise ValueError("ProteinBatch can only have 3 dimensions -- you tried to select " + str(len(idx)))
            if len(idx) == 3 and idx[2] != slice(None):
                raise ValueError("Cannot slice on sequence dimension of a ProteinBatch")
            return ProteinBatch(self.data[idx], self.batch_offsets, self.batch_lengths)
        
        raise ValueError("Cannot index on the batch index of a ProteinBatch: " + repr(idx))
        
    def __setitem__(self, idx, val):
        if isinstance(idx, slice):
            raise ValueError("Cannot slice on the batch index of a ProteinBatch")
        elif isinstance(idx, tuple):
            if idx[0] != slice(None):
                raise ValueError("Cannot slice on the batch index of a ProteinBatch")
            if len(idx) > 3:
                raise ValueError("ProteinBatch can only have 3 dimensions -- you tried to select " + str(len(idx)))
            if len(idx) == 3 and idx[2] != slice(None):
                raise ValueError("Cannot slice on sequence dimension of a ProteinBatch")
            
            if isinstance(val, ProteinBatch):
                self.data[idx] = val.data
            else:
                self.data[idx] = val
            return
        
        raise ValueError("Cannot index on the batch index of a ProteinBatch: " + repr(idx))

    def to(self, *args, **kwargs):
        return ProteinBatch(self._data.to(*args, **kwargs), self._batch_offsets, self._batch_lengths)
        
        
class ApplyToProteinBatch(nn.Module):
    def __init__(self, module_or_func):
        super().__init__()
        self.module_or_func = module_or_func
        
    def forward(self, batch, *args, **kwargs):
        if not isinstance(batch, ProteinBatch):
            raise ValueError("Expected ProteinBatch, got " + str(batch.__class__))
        result = self.module_or_func(batch.data, *args, **kwargs)
        return ProteinBatch(result, batch.batch_offsets, batch.batch_lengths)
        
        
class ApplyReducingToProteinBatch(nn.Module):
    def __init__(self, 
                 module_or_func: typ.Callable, 
                 expected_reduction=None,
                 ceil_mode=False
                ):
        super().__init__()
        
        self.module_or_func = module_or_func
        self._ceil_mode = ceil_mode
        
    def forward(self, batch, *args, **kwargs):
        if not isinstance(batch, ProteinBatch):
            raise ValueError("Expected ProteinBatch, got " + str(batch.__class__))
        result = self.module_or_func(batch.data, *args, **kwargs)
        raise NotImplemented()

        
class ApplyConvToProteinBatch(nn.Module):
    def __init__(self,
                 module_or_func: typ.Callable,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1
                ):
        super().__init__()
        
        self.module_or_func = module_or_func
        
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[-1]
        if isinstance(stride, tuple):
            stride = stride[-1]
        if isinstance(padding, tuple):
            padding = padding[-1]
        if isinstance(dilation, tuple):
            dilation = dilation[-1]
            
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = False
        
    def _build_result(self, batch, result):
        trunc = math.ceil if self.ceil_mode else math.floor
        
        lengths = [
            trunc((l + 2*self.padding - self.dilation * (self.kernel - 1) - 1)/self.stride + 1) 
            for l in batch.batch_lengths
        ]
        
        off_the_left = math.floor((self.dilation * (self.kernel - 1)/2 - self.padding) / self.stride)
        
        offsets = [math.floor(i/self.stride) - off_the_left for i in batch.batch_offsets]
        
        result = ProteinBatch(result, offsets, lengths)
        return result
        
    def forward(self, batch, *args, **kwargs):
        if not isinstance(batch, ProteinBatch):
            raise ValueError("Expected ProteinBatch, got " + str(batch.__class__))
        
        padding = min(
            batch.batch_offsets[i+1] - (batch.batch_offsets[i] + batch.batch_lengths[i])
            for i in range(len(batch.batch_lengths)-1)
        )
        
        if (self.kernel - 1)/2 > padding:
            result = [
                self.module_or_func(batch.data[:,:,off:(off+leng)], *args, **kwargs)
                for off,leng in zip(batch.batch_offsets, batch.batch_lengths)
            ]
            indices = None
            if isinstance(result[0], tuple):
                result, indices = zip(*result)
                
            result = create_protein_batch(result)
            if indices is not None:
                return result, indices
            return result

        result = self.module_or_func(batch.data, *args, **kwargs)
        return self._build_result(batch, result)
        
    
class ApplyPoolToProteinBatch(ApplyConvToProteinBatch):
    def __init__(self,
                 module_or_func: typ.Callable,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 ceil_mode=False
                ):
        super().__init__(module_or_func, kernel_size, stride or kernel_size, padding, dilation)
        self.ceil_mode = ceil_mode
        
        
def apply_to_protein_batch(module):
    if isinstance(module, _ConvNd):
        return ApplyConvToProteinBatch(module, module.kernel_size, module.stride, module.padding, module.dilation)
    elif all(hasattr(module, x) for x in POOL_ATTRIBUTES): 
        # quacks like a pool
        dilation = 1
        if hasattr(module, 'dilation'):
            dilation = module.dilation
            
        return ApplyPoolToProteinBatch(module, module.kernel_size, module.stride, module.padding, dilation, module.ceil_mode)
    return ApplyToProteinBatch(module)


def protein_batch_to_padded_batch(pbatch):
    if not isinstance(pbatch, ProteinBatch):
        raise ValueError("Expected ProteinBatch, got " + str(pbatch.__class__))
        
    max_size = max(pbatch.batch_lengths)
    
    pieces = [
        pbatch.data[:,:,start:(start+length)]
        for start, length in zip(pbatch.batch_offsets, pbatch.batch_lengths)
    ]
    
    pieces = [
        p if p.shape[-1] == max_size else F.pad(p, (0,max_size-p.shape[-1]))
        for p in pieces
    ]
    
    return T.cat(pieces, 0)


class ProteinBatchToPaddedBatch(nn.Module):
    def forward(self, x):
        return protein_batch_to_padded_batch(x)