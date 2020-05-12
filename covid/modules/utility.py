import torch as T
from torch import nn
from ..functions import silu
import numpy as np

__ALL__ = ['SiLU', 'NONLINEARITIES', 'Squeeze', 'ResidualBlock', 'ScaleSafeBatchNorm1d']

class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)

NONLINEARITIES = {
    'relu': nn.ReLU, 
    'silu': SiLU, 
    'sigmoid': nn.Sigmoid, 
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU
}

class Squeeze(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis=axis
        
    def __call__(self, x):
        return x.squeeze(self.axis)

class ResidualBlock(nn.Module):
    def __init__(self, block, nonlinearity='silu'):
        super().__init__()
        self.model_layer = block
        
        if isinstance(nonlinearity, nn.Module):
            self._nonlinearity = nonlinearity
        elif nonlinearity is None:
            self._nonlinearity = None
        else:
            self._nonlinearity = NONLINEARITIES[nonlinearity]()
        
    def forward(self, x, *args, **kwargs):
        act = self._nonlinearity if self._nonlinearity is not None else (lambda x: x)

        return x + act(self.model_layer(act(x), *args, **kwargs))

class ScaleSafeBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, *args, burn_in=100, **kwargs):
        super().__init__(num_features, *args, **kwargs)

        self.register_buffer('is_log', T.zeros(num_features))

        self.register_buffer('bucket_counts', T.ones(13, num_features))
        self.register_buffer('log_bucket_counts', T.ones(13, num_features))

        self.register_buffer('running_lin_var', T.ones(1, num_features))
        self.register_buffer('running_log_var', T.ones(1, num_features))

        self.register_buffer('running_lin_mean', T.zeros(1, num_features))
        self.register_buffer('running_log_mean', T.zeros(1, num_features))

        # Normal dist for comparison
        self.register_buffer(
            'buckets_ref', 
            T.tensor(
                [0.003, 0.0092, 0.028, 0.066, 0.121, 0.175, 0.197, 0.175, 0.121, 0.066, 0.028, 0.0092, 0.003]
            ).unsqueeze(-1)
        )

        # Half way between normal and uniform
        self.buckets_ref = 0.5 * (self.buckets_ref + (1/13))

        self.burn_in_left = burn_in

    def forward(self, x):
        running_sd = T.sqrt(self.running_var)

        x_log = T.log(T.clamp(x, min=0) + 1.0) - T.log(T.clamp(-x, min=0) + 1.0)
        x_final = x_log * self.is_log + x * (1 - self.is_log)

        x_clamp = T.max(x_final, self.running_mean - running_sd * 5)
        x_clamp = T.min(x_clamp, self.running_mean + running_sd * 5)

        x_pos_ln = T.log(T.clamp(x_final - (self.running_mean + running_sd * 5), min=0) + 1)
        x_neg_ln = T.log(T.clamp(self.running_mean - running_sd * 5 - x_final, min=0) + 1)

        x_clamp = x_clamp + x_pos_ln - x_neg_ln
        
        result = super().forward(x_clamp)

        if self.burn_in_left > 0 and self.training:
            self.running_lin_mean += self.momentum * (x.detach().mean(0, keepdim=True) - self.running_lin_mean)
            self.running_log_mean += self.momentum * (x_log.detach().mean(0, keepdim=True) - self.running_log_mean)
            self.running_lin_var += self.momentum * ((x.detach()-self.running_lin_mean).var(0, keepdim=True) - self.running_lin_var)
            self.running_log_var += self.momentum * ((x_log.detach()-self.running_log_mean).var(0, keepdim=True) - self.running_log_var)
        
            z_scores = (x.detach() - self.running_lin_mean) / T.sqrt(self.running_lin_var)
            z_log_scores = (x_log.detach() - self.running_log_mean) / T.sqrt(self.running_log_var)

            bucket_update = T.zeros_like(self.bucket_counts)
            log_bucket_update = T.zeros_like(self.log_bucket_counts)

            for i, cutoff in enumerate(np.linspace(-2.75, 2.75, 12)):
                bucket_update[i] += (z_scores <= cutoff).sum(0)
                log_bucket_update[i] += (z_log_scores <= cutoff).sum(0)

            bucket_update[-1] += T.isfinite(z_scores).sum(0)
            log_bucket_update[-1] += T.isfinite(z_log_scores).sum(0)

            bucket_update[1:] = bucket_update[1:] - bucket_update[:-1]
            log_bucket_update[1:] = log_bucket_update[1:] - log_bucket_update[:-1]

            self.bucket_counts += bucket_update
            self.log_bucket_counts += log_bucket_update

            p = self.bucket_counts/self.bucket_counts.sum(0,keepdim=True)
            p_log = self.log_bucket_counts/self.log_bucket_counts.sum(0,keepdim=True)

            lin_chisq = (((p-self.buckets_ref)**2)/self.buckets_ref).sum(0)
            log_chisq = (((p_log-self.buckets_ref)**2)/self.buckets_ref).sum(0)

            updated_selection = 1.0 * (log_chisq < lin_chisq)
            self.is_log = self.is_log + self.momentum * (updated_selection - self.is_log)

            self.burn_in_left -= 1
            if self.burn_in_left == 0:
                self.is_log = T.round(self.is_log)
            

        return result