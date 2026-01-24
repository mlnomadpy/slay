"""
Novel activation functions for Yat-style attention.
"""

import torch


def novel_activation(x, w_out, eps=1e-4):
    """
    Computes (x·w)² / (||x - w||² + eps).
    Executes in FP32 to prevent NaN/Inf in FP16 training.
    """
    input_dtype = x.dtype
    
    # UPCAST TO FP32 FOR NUMERICAL STABILITY
    x = x.float()
    w_out = w_out.float()
    
    if x.shape[-1] != w_out.shape[-1]:
        D_in = x.shape[-1]
        x_sum = x.sum(dim=-1, keepdim=True)
        # Safe squaring in FP32
        x_sq_sum = (x.pow(2)).sum(dim=-1, keepdim=True)
        
        # Expansion: ||x - w||^2
        dist_sq = x_sq_sum - (2 * w_out * x_sum) + (D_in * w_out.pow(2)) + eps
        dist_sq = torch.relu(dist_sq)  # Clamp negatives from precision error
        
        activation = 1.0 / (1.0 + dist_sq.sqrt())
        out = w_out * activation
    else:
        dot_product = (x * w_out).sum(dim=-1, keepdim=True)
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        w_norm_sq = (w_out * w_out).sum(dim=-1, keepdim=True)
        
        distance_sq = x_norm_sq + w_norm_sq - 2 * dot_product + eps
        numerator = dot_product ** 2
        activation_scale = numerator / distance_sq
        activation_scale = torch.clamp(activation_scale, 0, 10.0)
        out = w_out * activation_scale

    # CAST BACK TO ORIGINAL DTYPE (FP16)
    return out.to(input_dtype)
