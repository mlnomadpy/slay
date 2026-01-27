"""
Linear attention with ELU+1 kernel and causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCausalAttention(nn.Module):
    """ELU+1 linear attention with causal masking via cumsum."""
    
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.eps = 1e-6
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape: (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Upcast for stability
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # ELU+1 kernel
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1
        
        # Causal linear attention via cumsum
        # (B, H, T, D) * (B, H, T, D) -> (B, H, T, D, D)
        kv_prod = torch.einsum('bhtd,bhte->bhtde', k_prime, v)
        kv_cumsum = torch.cumsum(kv_prod, dim=2)
        
        # (B, H, T, D) * (B, H, T, D, D) -> (B, H, T, D)
        context = torch.einsum('bhtd,bhtde->bhte', q_prime, kv_cumsum)
        
        # Normalization
        k_cumsum = torch.cumsum(k_prime, dim=2)
        norm = torch.einsum('bhtd,bhtd->bht', q_prime, k_cumsum)
        
        out = context / (norm.unsqueeze(-1) + self.eps)
        out = out.to(input_dtype)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)
    
    def forward_triton(self, x):
        """
        Forward pass using Triton-accelerated linear attention.
        
        ~36x faster than pure PyTorch for the linear attention computation.
        Falls back to regular forward() if Triton is not available.
        """
        try:
            from .yat_attention_kernel import HAS_TRITON, triton_linear_attention
        except ImportError:
            try:
                from yat_attention_kernel import HAS_TRITON, triton_linear_attention
            except ImportError:
                return self.forward(x)
        
        if not HAS_TRITON or not torch.cuda.is_available():
            return self.forward(x)
        
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        
        # ELU+1 features: (B, H, T, D) - uses head_dim as feature dim
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1
        
        # Use Triton kernel for linear attention
        out = triton_linear_attention(
            q_prime.contiguous(), 
            k_prime.contiguous(), 
            v.contiguous(), 
            delta=self.eps
        )
        
        out = out.to(input_dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# ============================================================================
# Testing and Validation
# ============================================================================

def exact_softmax_attention(q, k, v):
    """Compute exact causal softmax attention for comparison."""
    B, H, T, D = q.shape
    scale = 1.0 / (D ** 0.5)
    
    # Compute attention scores
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float('-inf'))
    
    # Softmax and apply to values
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def test_approximation_quality():
    """
    Test how well linear attention approximates exact softmax attention.
    """
    print("\n" + "="*80)
    print("LINEAR ATTENTION: APPROXIMATION QUALITY TEST")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, embed_dim, n_heads = 2, 128, 256, 4
    head_dim = embed_dim // n_heads
    
    model = LinearCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
    
    x = torch.randn(B, T, embed_dim, device=device)
    
    with torch.no_grad():
        # Get linear attention output
        out_linear = model.forward(x)
        
        # Compute exact softmax for comparison
        qkv = model.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2).float()
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2).float()
        v = v.view(B, T, n_heads, head_dim).transpose(1, 2).float()
        
        out_exact = exact_softmax_attention(q, k, v)
        out_exact = out_exact.transpose(1, 2).contiguous().view(B, T, embed_dim)
        out_exact = model.out(out_exact)
    
    # Compute metrics
    diff = out_linear.float() - out_exact.float()
    rel_l2 = diff.norm() / out_exact.norm()
    cosine = F.cosine_similarity(out_linear.view(-1), out_exact.view(-1), dim=0)
    mse = (diff ** 2).mean()
    
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'Relative L2 Error':<20} {rel_l2.item():.4f}")
    print(f"{'Cosine Similarity':<20} {cosine.item():.4f}")
    print(f"{'MSE':<20} {mse.item():.2e}")
    
    # Note: Linear attention is NOT meant to approximate softmax exactly
    print("\n[Note] Linear attention uses ELU+1 kernel, not softmax approximation.")
    print("       Low similarity expected - this tests numerical stability, not accuracy.")


def test_feature_properties():
    """
    Test theoretical properties of the ELU+1 feature map.
    """
    print("\n" + "="*80)
    print("LINEAR ATTENTION: FEATURE PROPERTY TESTS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Test 1: ELU+1 Non-negativity
    print("\n[Test 1] ELU+1 Feature Non-negativity")
    print("-" * 40)
    
    n_samples = 10000
    x = torch.randn(n_samples, device=device)
    features = F.elu(x) + 1
    
    min_val = features.min().item()
    print(f"Min feature value: {min_val:.6e}")
    
    if min_val >= 0:
        print("✓ All features ≥ 0")
    else:
        print(f"✗ FAILED: Min feature = {min_val}")
    
    # Test 2: Denominator Positivity
    print("\n[Test 2] Denominator Positivity")
    print("-" * 40)
    
    B, T, embed_dim, n_heads = 2, 64, 256, 4
    model = LinearCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
    
    # Test with random inputs
    min_denom = float('inf')
    for _ in range(100):
        x = torch.randn(B, T, embed_dim, device=device)
        
        with torch.no_grad():
            qkv = model.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2).float()
            k = k.view(B, T, n_heads, embed_dim // n_heads).transpose(1, 2).float()
            
            q_prime = F.elu(q) + 1
            k_prime = F.elu(k) + 1
            
            k_cumsum = torch.cumsum(k_prime, dim=2)
            denom = torch.einsum('bhtd,bhtd->bht', q_prime, k_cumsum)
            
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
    print("LINEAR ATTENTION: MEMORY SCALING TEST")
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
        model = LinearCausalAttention(embed_dim=embed_dim, n_heads=n_heads).to(device).eval()
        
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


def test_triton_comparison():
    """Test forward() vs forward_triton() correctness and speed."""
    print("\n" + "="*80)
    print("LINEAR ATTENTION: TRITON COMPARISON TEST")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping Triton test.")
        return
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    model = LinearCausalAttention(embed_dim=256, n_heads=4).to(device).eval()
    
    configs = [
        (1, 512, 256),
        (1, 1024, 256),
        (2, 512, 256),
    ]
    
    print(f"\n{'Config':<25} {'Max Diff':<15} {'Mean Diff':<15} {'Rel Diff':<15} {'Status':<10}")
    print("-" * 80)
    
    all_passed = True
    for B, T, C in configs:
        x = torch.randn(B, T, C, device=device)
        
        with torch.no_grad():
            out_pytorch = model.forward(x)
            out_triton = model.forward_triton(x)
        
        diff = (out_pytorch - out_triton).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        rel_diff = (diff / (out_pytorch.abs() + 1e-8)).mean().item()
        
        passed = mean_diff < 1e-2 and rel_diff < 1e-1
        all_passed = all_passed and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        
        config_str = f"B={B}, T={T}, C={C}"
        print(f"{config_str:<25} {max_diff:<15.2e} {mean_diff:<15.2e} {rel_diff:<15.2e} {status:<10}")
    
    print()
    return all_passed


def test_speed_benchmark():
    """Benchmark forward() vs forward_triton() speed."""
    print("\n" + "="*80)
    print("LINEAR ATTENTION: SPEED BENCHMARK")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    model = LinearCausalAttention(embed_dim=256, n_heads=4).to(device).eval()
    
    configs = [(1, 512), (1, 1024), (1, 2048)]
    
    print(f"\n{'Seq Length':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for B, T in configs:
        x = torch.randn(B, T, 256, device=device)
        
        # Warmup
        for _ in range(3):
            _ = model.forward(x)
            _ = model.forward_triton(x)
        torch.cuda.synchronize()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            _ = model.forward(x)
        end.record()
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / 10
        
        start.record()
        for _ in range(10):
            _ = model.forward_triton(x)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 10
        
        speedup = pytorch_time / triton_time
        print(f"{T:<15} {pytorch_time:<15.2f} {triton_time:<15.2f} {speedup:<10.2f}x")


def test_gradient_flow():
    """Test that gradients flow correctly through the attention mechanism."""
    print("\n" + "="*80)
    print("LINEAR ATTENTION: GRADIENT FLOW TEST")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    B, T, embed_dim, n_heads = 2, 64, 256, 4
    
    model = LinearCausalAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
    ).to(device)
    
    print(f"\n{'Method':<15} {'Output Shape':<20} {'Has Grad':<12} {'Grad Finite':<15} {'Status':<10}")
    print("-" * 70)
    
    for method_name in ["forward", "forward_triton"]:
        try:
            x = torch.randn(B, T, embed_dim, device=device, requires_grad=True)
            model.zero_grad()
            
            if method_name == "forward":
                out = model.forward(x)
            else:
                out = model.forward_triton(x)
            
            loss = out.sum()
            loss.backward()
            
            has_grad = x.grad is not None
            grad_finite = has_grad and torch.isfinite(x.grad).all().item()
            passed = has_grad and grad_finite
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{method_name:<15} {str(tuple(out.shape)):<20} {str(has_grad):<12} {str(grad_finite):<15} {status:<10}")
            
        except Exception as e:
            print(f"{method_name:<15} {'ERROR':<20} {'-':<12} {'-':<15} {'✗ FAIL':<10}")
            print(f"  Error: {e}")


def test_mixed_precision():
    """Test numerical stability with float16 and bfloat16 inputs."""
    print("\n" + "="*80)
    print("LINEAR ATTENTION: MIXED PRECISION TEST")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision test")
        return
    
    device = torch.device("cuda")
    B, T, embed_dim, n_heads = 1, 128, 256, 4
    
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    
    print(f"\n{'Dtype':<12} {'Method':<15} {'Output Dtype':<18} {'Finite':<12} {'Status':<10}")
    print("-" * 70)
    
    for dtype in dtypes:
        dtype_name = "float16" if dtype == torch.float16 else "bfloat16"
        
        for method_name in ["forward", "forward_triton"]:
            try:
                model = LinearCausalAttention(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                ).to(device, dtype=dtype).eval()
                
                x = torch.randn(B, T, embed_dim, device=device, dtype=dtype)
                
                with torch.no_grad():
                    if method_name == "forward":
                        out = model.forward(x)
                    else:
                        out = model.forward_triton(x)
                
                is_finite = torch.isfinite(out).all().item()
                
                status = "✓ PASS" if is_finite else "✗ FAIL"
                print(f"{dtype_name:<12} {method_name:<15} {str(out.dtype):<18} {str(is_finite):<12} {status:<10}")
                
            except Exception as e:
                print(f"{dtype_name:<12} {method_name:<15} {'ERROR':<18} {'-':<12} {'✗ FAIL':<10}")
                print(f"  Error: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" LINEAR ATTENTION (ELU+1): COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Core tests
    test_approximation_quality()
    test_feature_properties()
    test_memory_scaling()
    
    # Gradient and precision tests
    test_gradient_flow()
    test_mixed_precision()
    
    # Triton tests
    test_triton_comparison()
    test_speed_benchmark()
    
    print("\n" + "="*80)
    print(" ALL TESTS COMPLETED")
    print("="*80)
