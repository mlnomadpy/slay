"""
Yat-product attention (exact) with causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YatCausalAttention(nn.Module):
    """Yat-product attention (exact) with causal masking.
    
    Uses kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
    
    Args:
        embed_dim: Embedding dimension
        n_heads: Number of attention heads
        epsilon: Small constant for numerical stability
        score_scale: Scale factor for attention scores (default: sqrt(2))
    """
    
    def __init__(self, embed_dim, n_heads, epsilon=1e-6, score_scale=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.epsilon = epsilon
        
        # Default scale is 1, stored as tensor for precision
        if score_scale is None:
            self.register_buffer('score_scale', torch.tensor(1.0))
        else:
            self.register_buffer('score_scale', torch.tensor(float(score_scale)).sqrt())
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape: (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute dot product: q·k
        dot_product = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        
        # Compute squared norms
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        
        # Yat kernel: (q·k)² / (||q||² + ||k||² - 2*q·k + eps)
        # ||q||² + ||k||² broadcasted: (B, H, T, 1) + (B, H, 1, T) -> (B, H, T, T)
        numerator = dot_product ** 2
        denominator = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot_product + self.epsilon
        
        scores = numerator / denominator
        
        # Apply score scaling (uses tensor for correct dtype/device)
        scores = scores * self.score_scale.to(scores.dtype)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
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
    Test how well Yat attention approximates exact softmax.
    """
    print("\n" + "="*80)
    print("YAT ATTENTION: APPROXIMATION QUALITY TEST")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, embed_dim, n_heads = 2, 128, 256, 4
    head_dim = embed_dim // n_heads
    
    configs = [
        ("epsilon=1e-6", 1e-6),
        ("epsilon=1e-4", 1e-4),
        ("epsilon=1e-2", 1e-2),
    ]
    
    print(f"\n{'Config':<20} {'Rel L2↓':<12} {'Cosine↑':<12} {'MSE↓':<12}")
    print("-" * 56)
    
    for config_name, epsilon in configs:
        model = YatCausalAttention(
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
    
    print("\n[Note] Yat uses (q·k)²/(||q-k||²+ε) kernel with softmax normalization.")
    print("       Different from standard scaled dot-product attention.")


def test_feature_properties():
    """
    Test theoretical properties of Yat kernel.
    """
    print("\n" + "="*80)
    print("YAT ATTENTION: FEATURE PROPERTY TESTS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Test 1: Kernel Non-negativity ((q·k)² always ≥ 0)
    print("\n[Test 1] Kernel Non-negativity")
    print("-" * 40)
    
    B, T, embed_dim, n_heads = 2, 64, 256, 4
    model = YatCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
    
    min_score = float('inf')
    for _ in range(100):
        x = torch.randn(B, T, embed_dim, device=device)
        
        with torch.no_grad():
            qkv = model.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2)
            k = k.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2)
            
            dot_product = torch.matmul(q, k.transpose(-2, -1))
            q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
            k_norm_sq = (k * k).sum(dim=-1, keepdim=True)
            
            numerator = dot_product ** 2
            denominator = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot_product + model.epsilon
            scores = numerator / denominator
            
            min_score = min(min_score, scores.min().item())
    
    print(f"Min kernel value: {min_score:.6e}")
    
    if min_score >= 0:
        print("✓ All kernel values ≥ 0")
    else:
        print(f"✗ FAILED: Min kernel = {min_score}")
    
    # Test 2: Denominator Positivity
    print("\n[Test 2] Denominator Positivity (||q-k||² + ε > 0)")
    print("-" * 40)
    
    min_denom = float('inf')
    for _ in range(100):
        x = torch.randn(B, T, embed_dim, device=device)
        
        with torch.no_grad():
            qkv = model.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2)
            k = k.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2)
            
            dot_product = torch.matmul(q, k.transpose(-2, -1))
            q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
            k_norm_sq = (k * k).sum(dim=-1, keepdim=True)
            
            denominator = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot_product + model.epsilon
            min_denom = min(min_denom, denominator.min().item())
    
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
    print("YAT ATTENTION: MEMORY SCALING TEST")
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
        model = YatCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
        
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
    print("YAT ATTENTION: GRADIENT FLOW TEST")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, embed_dim, n_heads = 2, 64, 256, 4
    
    model = YatCausalAttention(
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
    print("YAT ATTENTION: MIXED PRECISION TEST")
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
            model = YatCausalAttention(
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
    print(" YAT ATTENTION: COMPREHENSIVE TEST SUITE")
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
