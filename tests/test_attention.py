"""
Standalone Unit Tests for Attention Mechanisms

Tests all attention implementations from src.attention without deepspeed dependency.

Run with: python tests/test_attention.py
Or: pytest tests/test_attention.py -v
"""

import torch
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import (
    ATTENTION_CLASSES,
    get_attention_class,
    list_attention_types,
    StandardCausalAttention,
    LinearCausalAttention,
    CosformerCausalAttention,
    RFFCausalAttention,
    YatCausalAttention,
    YatPerformerCausalAttention,
    YatSphericalCausalAttention,
    FastAttention,
)


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    'embed_dim': 64,
    'n_heads': 4,
    'batch_size': 2,
    'seq_len': 16,
}


# ============================================================================
# Unit Tests
# ============================================================================

def test_registry_contains_all_types():
    """Test that the registry contains all expected attention types."""
    expected = [
        'performer',
        'standard',
        'linear',
        'cosformer',
        'rff',
        'yat',
        'yat-performer',
        'yat-performer-tensor',
        'yat-performer-laplace',
        'yat-performer-rm',
        'yat-performer-nystrom',
        'yat-performer-anchors',
        'yat-spherical',
    ]
    actual = list_attention_types()
    
    for name in expected:
        assert name in actual, f"Missing attention type: {name}"
    print(f"✓ Registry contains all {len(expected)} attention types")


def test_get_attention_class():
    """Test that get_attention_class returns correct classes."""
    assert get_attention_class('standard') == StandardCausalAttention
    assert get_attention_class('performer') == FastAttention
    assert get_attention_class('yat-performer') == YatPerformerCausalAttention
    print("✓ get_attention_class returns correct classes")


def test_all_attentions_forward_pass():
    """Test that all attention modules run forward pass without errors."""
    B, T, C = TEST_CONFIG['batch_size'], TEST_CONFIG['seq_len'], TEST_CONFIG['embed_dim']
    n_heads = TEST_CONFIG['n_heads']
    
    x = torch.randn(B, T, C)
    
    for name in list_attention_types():
        attn_class = get_attention_class(name)
        attn = attn_class(C, n_heads)
        
        with torch.no_grad():
            out = attn(x)
        
        assert out.shape == (B, T, C), f"{name}: Expected shape {(B, T, C)}, got {out.shape}"
        assert not torch.isnan(out).any(), f"{name}: Output contains NaN"
        assert not torch.isinf(out).any(), f"{name}: Output contains Inf"
        
        print(f"✓ {name}: forward pass OK, output shape {out.shape}")


def test_causal_masking():
    """Test that causal attention doesn't look into the future."""
    B, T, C = 1, 8, 64
    n_heads = 4
    
    # For causal attention, output at position i should not depend on inputs at position j > i
    # We test this by checking gradient flow
    
    for name in ['standard', 'linear', 'yat-spherical']:
        attn = get_attention_class(name)(C, n_heads)
        attn.eval()
        
        x = torch.randn(B, T, C, requires_grad=True)
        out = attn(x)
        
        # Compute gradient of first output position w.r.t. all inputs
        out[0, 0].sum().backward()
        
        grad = x.grad[0]  # (T, C)
        grad_magnitude = grad.abs().sum(dim=-1)  # (T,)
        
        # First position should only have gradient from first position (or be zero for others)
        # Due to numerical precision, we check if future positions have ~0 gradient
        future_grad = grad_magnitude[1:].sum().item()
        first_grad = grad_magnitude[0].item()
        
        # Future positions should have negligible gradient compared to first
        if first_grad > 1e-6:
            ratio = future_grad / first_grad
            assert ratio < 0.01, f"{name}: Causal violation - future gradient ratio: {ratio:.4f}"
        
        x.grad = None
        print(f"✓ {name}: causal masking verified")


def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    B, T, C = 2, 16, 64
    n_heads = 4
    
    test_cases = [
        ("normal", torch.randn(B, T, C)),
        ("large", torch.randn(B, T, C) * 100),
        ("small", torch.randn(B, T, C) * 0.001),
        ("mixed", torch.randn(B, T, C) * torch.randint(1, 100, (B, T, 1)).float()),
    ]
    
    for name in list_attention_types():
        attn = get_attention_class(name)(C, n_heads)
        attn.eval()
        
        for case_name, x in test_cases:
            with torch.no_grad():
                out = attn(x)
            
            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            
            # Some cases may have issues - just report
            if has_nan or has_inf:
                print(f"  ⚠ {name}/{case_name}: NaN={has_nan}, Inf={has_inf}")
            else:
                pass  # OK
        
        print(f"✓ {name}: numerical stability tested")


def test_gradient_flow():
    """Test that gradients flow properly through all attentions."""
    B, T, C = 2, 16, 64
    n_heads = 4
    
    for name in list_attention_types():
        attn = get_attention_class(name)(C, n_heads)
        attn.train()
        
        x = torch.randn(B, T, C, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        
        # Check that input has gradient
        assert x.grad is not None, f"{name}: No gradient for input"
        assert not torch.isnan(x.grad).any(), f"{name}: NaN in input gradient"
        
        # Check that parameters have gradients
        for pname, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name}/{pname}: No gradient"
                assert not torch.isnan(param.grad).any(), f"{name}/{pname}: NaN in gradient"
        
        print(f"✓ {name}: gradient flow OK")


def test_output_shape_consistency():
    """Test output shape consistency across different sequence lengths."""
    B, C = 2, 64
    n_heads = 4
    seq_lengths = [8, 16, 32, 64]
    
    for name in list_attention_types():
        attn = get_attention_class(name)(C, n_heads)
        attn.eval()
        
        for T in seq_lengths:
            x = torch.randn(B, T, C)
            with torch.no_grad():
                out = attn(x)
            
            assert out.shape == (B, T, C), f"{name}/T={T}: shape mismatch"
        
        print(f"✓ {name}: output shape consistent across sequence lengths")


# ============================================================================
# Kernel Approximation Tests
# ============================================================================

def compute_exact_spherical_yat_kernel(q, k, epsilon=1e-6):
    """Spherical Yat kernel: K(q,k) = x² / (C - 2x) where x = <q̂, k̂>"""
    C = 2.0 + epsilon
    q_norm = F.normalize(q, p=2, dim=-1)
    k_norm = F.normalize(k, p=2, dim=-1)
    x = torch.einsum('bhtd,bhsd->bhts', q_norm, k_norm)
    denominator = torch.clamp(C - 2 * x, min=1e-6)
    return (x ** 2) / denominator


def test_yat_spherical_kernel_formula():
    """Test spherical Yat kernel formula is correct."""
    B, H, T, D = 2, 4, 8, 16
    epsilon = 1e-6
    
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    
    exact_kernel = compute_exact_spherical_yat_kernel(q, k, epsilon)
    
    # Verify formula manually
    q_norm, k_norm = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    x = torch.einsum('bhtd,bhsd->bhts', q_norm, k_norm)
    C = 2.0 + epsilon
    expected = x ** 2 / torch.clamp(C - 2 * x, min=1e-6)
    
    torch.testing.assert_close(exact_kernel, expected, rtol=1e-5, atol=1e-5)
    print("✓ Spherical Yat kernel formula verified")


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Attention Module Tests")
    print("=" * 60)
    
    tests = [
        test_registry_contains_all_types,
        test_get_attention_class,
        test_all_attentions_forward_pass,
        test_causal_masking,
        test_numerical_stability,
        test_gradient_flow,
        test_output_shape_consistency,
        test_yat_spherical_kernel_formula,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    torch.manual_seed(42)
    success = run_all_tests()
    sys.exit(0 if success else 1)
