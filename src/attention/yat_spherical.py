"""
Exact Spherical Yat attention with causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YatSphericalCausalAttention(nn.Module):
    """Exact spherical Yat attention with causal masking.

    Uses kernel:
        K(q,k) = x^2 / (C - 2x),  where x = <q̂, k̂> and C = 2 + ε.

    This module implements *kernel-normalized* attention (linear-attention style):
        Y = (K V) / (K 1)
    under a causal mask, i.e. it does not apply a softmax.

    FP32 upcast added for numerical stability in mixed-precision training.

    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        epsilon: Small constant for numerical stability
        score_scale: Optional multiplicative factor applied to K (cancels under K-normalization)
    """
    
    def __init__(self, embed_dim, n_heads, epsilon=1e-2, score_scale=None, **kwargs):
        super().__init__()
        
        # Handle parameter aliases
        if 'num_quadrature_nodes' in kwargs: 
             # Just consume it, spherical (exact) doesn't use quadrature but might receive it
             pass
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        self.C = 2.0 + epsilon  # Larger epsilon = more numerical headroom
        
        # Default scale is sqrt(2), stored as tensor for precision
        if score_scale is None:
            self.register_buffer('score_scale', torch.tensor(1.0))
        else:
            self.register_buffer('score_scale', torch.tensor(float(score_scale)))
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        B, T, C_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # UPCAST TO FP32 FOR NUMERICAL STABILITY
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # Normalize to unit sphere
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # x = <q̂, k̂> ∈ [-1, 1]
        x_dot = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        
        # Kernel: x² / (C - 2x)
        denom = torch.clamp(self.C - 2 * x_dot, min=1e-6)
        K = (x_dot ** 2) / denom

        # Optional scaling (cancels under kernel normalization)
        K = K * self.score_scale.to(K.dtype)

        # Causal mask: zero out future contributions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        K = K.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        # Kernel-normalized attention: (K V) / (K 1)
        numerator = torch.matmul(K, v)  # (B, H, T, D)
        denominator = K.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        out = numerator / (denominator + 1e-6)
        
        # CAST BACK TO ORIGINAL DTYPE
        out = out.to(input_dtype)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C_dim)
        return self.out(out)


# ============================================================================
# Testing and Validation
# ============================================================================

def exact_softmax_attention(q, k, v):
    """Compute exact causal softmax attention for comparison."""
    B, H, T, D = q.shape
    scale = 1.0 / (D ** 0.5)
    
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def test_approximation_quality():
    """
    Test how well spherical Yat attention approximates exact softmax.
    """
    print("\n" + "="*80)
    print("YAT SPHERICAL: APPROXIMATION QUALITY TEST")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, embed_dim, n_heads = 2, 128, 256, 4
    head_dim = embed_dim // n_heads
    
    configs = [
        ("epsilon=1e-2", 1e-2),
        ("epsilon=1e-1", 1e-1),
        ("epsilon=0.5", 0.5),
    ]
    
    print(f"\n{'Config':<20} {'Rel L2↓':<12} {'Cosine↑':<12} {'MSE↓':<12}")
    print("-" * 56)
    
    for config_name, epsilon in configs:
        model = YatSphericalCausalAttention(
            embed_dim=embed_dim, n_heads=n_heads, epsilon=epsilon
        ).to(device).eval()
        
        x = torch.randn(B, T, embed_dim, device=device)
        
        with torch.no_grad():
            out_yat = model.forward(x)
            
            qkv = model.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, n_heads, head_dim).transpose(1, 2).float()
            k = k.view(B, T, n_heads, head_dim).transpose(1, 2).float()
            v = v.view(B, T, n_heads, head_dim).transpose(1, 2).float()
            
            out_exact = exact_softmax_attention(q, k, v)
            out_exact = out_exact.transpose(1, 2).contiguous().view(B, T, embed_dim)
            out_exact = model.out(out_exact)
        
        diff = out_yat.float() - out_exact.float()
        rel_l2 = diff.norm() / out_exact.norm()
        cosine = F.cosine_similarity(out_yat.view(-1), out_exact.view(-1), dim=0)
        mse = (diff ** 2).mean()
        
        print(f"{config_name:<20} {rel_l2.item():<12.4f} {cosine.item():<12.4f} {mse.item():<12.2e}")
    
    print("\n[Note] Yat spherical uses x²/(C-2x) kernel, not softmax.")
    print("       Different behavior expected - this tests numerical stability.")


def test_feature_properties():
    """
    Test theoretical properties of Yat spherical kernel.
    """
    print("\n" + "="*80)
    print("YAT SPHERICAL: FEATURE PROPERTY TESTS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Test 1: Kernel Non-negativity (x² / (C - 2x) ≥ 0 for x ∈ [-1, 1])
    print("\n[Test 1] Kernel Non-negativity")
    print("-" * 40)
    
    epsilon = 1e-2
    C = 2.0 + epsilon
    
    # Sample points in [-1, 1]
    x = torch.linspace(-1, 1, 1000, device=device)
    denom = torch.clamp(C - 2 * x, min=1e-6)
    K = (x ** 2) / denom
    
    min_val = K.min().item()
    print(f"Min kernel value: {min_val:.6e}")
    
    if min_val >= 0:
        print("✓ All kernel values ≥ 0")
    else:
        print(f"✗ FAILED: Min kernel = {min_val}")
    
    # Test 2: Denominator Positivity
    print("\n[Test 2] Denominator Positivity")
    print("-" * 40)
    
    B, T, embed_dim, n_heads = 2, 64, 256, 4
    model = YatSphericalCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
    
    min_denom = float('inf')
    for _ in range(100):
        x = torch.randn(B, T, embed_dim, device=device)
        
        with torch.no_grad():
            qkv = model.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2).float()
            k = k.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2).float()
            
            q_norm = F.normalize(q, p=2, dim=-1)
            k_norm = F.normalize(k, p=2, dim=-1)
            
            x_dot = torch.matmul(q_norm, k_norm.transpose(-2, -1))
            denom = model.C - 2 * x_dot
            
            min_denom = min(min_denom, denom.min().item())
    
    print(f"Min denominator: {min_denom:.6e}")
    
    if min_denom > 0:
        print("✓ All denominators > 0")
    else:
        print(f"✗ FAILED: Min denominator = {min_denom}")


def test_memory_scaling():
    """
    Test memory usage scaling with sequence length.
    """
    print("\n" + "="*80)
    print("YAT SPHERICAL: MEMORY SCALING TEST")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    embed_dim, n_heads = 256, 4
    seq_lengths = [512, 1024, 2048, 4096]
    
    print(f"\n{'Seq Length':<15} {'Peak Memory (MB)':<20} {'Status':<10}")
    print("-" * 45)
    
    for T in seq_lengths:
        model = YatSphericalCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(1, T, embed_dim, device=device)
        
        try:
            with torch.no_grad():
                _ = model.forward(x)
            
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"{T:<15} {peak_mem:<20.2f} {'✓':<10}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{T:<15} {'OOM':<20} {'✗':<10}")
                torch.cuda.empty_cache()
            else:
                raise
        
        del model
        torch.cuda.empty_cache()


def test_gradient_flow():
    """Test that gradients flow correctly through the attention mechanism."""
    print("\n" + "="*80)
    print("YAT SPHERICAL: GRADIENT FLOW TEST")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, embed_dim, n_heads = 2, 64, 256, 4
    
    model = YatSphericalCausalAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
    ).to(device)
    
    print(f"\n{'Method':<15} {'Output Shape':<20} {'Has Grad':<12} {'Grad Finite':<15} {'Status':<10}")
    print("-" * 70)
    
    try:
        x = torch.randn(B, T, embed_dim, device=device, requires_grad=True)
        model.zero_grad()
        
        out = model.forward(x)
        
        loss = out.sum()
        loss.backward()
        
        has_grad = x.grad is not None
        grad_finite = has_grad and torch.isfinite(x.grad).all().item()
        passed = has_grad and grad_finite
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{'forward':<15} {str(tuple(out.shape)):<20} {str(has_grad):<12} {str(grad_finite):<15} {status:<10}")
        
    except Exception as e:
        print(f"{'forward':<15} {'ERROR':<20} {'-':<12} {'-':<15} {'✗ FAIL':<10}")
        print(f"  Error: {e}")


def test_mixed_precision():
    """Test numerical stability with float16 and bfloat16 inputs."""
    print("\n" + "="*80)
    print("YAT SPHERICAL: MIXED PRECISION TEST")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision test")
        return
    
    device = torch.device("cuda")
    B, T, embed_dim, n_heads = 1, 128, 256, 4
    
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    
    print(f"\n{'Dtype':<12} {'Output Dtype':<18} {'Finite':<12} {'Status':<10}")
    print("-" * 55)
    
    for dtype in dtypes:
        dtype_name = "float16" if dtype == torch.float16 else "bfloat16"
        
        try:
            model = YatSphericalCausalAttention(
                embed_dim=embed_dim,
                n_heads=n_heads,
            ).to(device, dtype=dtype).eval()
            
            x = torch.randn(B, T, embed_dim, device=device, dtype=dtype)
            
            with torch.no_grad():
                out = model.forward(x)
            
            is_finite = torch.isfinite(out).all().item()
            
            status = "✓ PASS" if is_finite else "✗ FAIL"
            print(f"{dtype_name:<12} {str(out.dtype):<18} {str(is_finite):<12} {status:<10}")
            
        except Exception as e:
            print(f"{dtype_name:<12} {'ERROR':<18} {'-':<12} {'✗ FAIL':<10}")
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" YAT SPHERICAL ATTENTION: COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Core tests
    test_approximation_quality()
    test_feature_properties()
    test_memory_scaling()
    
    # Gradient and precision tests
    test_gradient_flow()
    test_mixed_precision()
    
    print("\n" + "="*80)
    print(" ALL TESTS COMPLETED")
    print("="*80)
