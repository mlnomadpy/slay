
import torch
import time
import pandas as pd
from src.attention.standard import StandardCausalAttention
from src.attention.cosformer import CosformerCausalAttention
from src.attention.yat_performer import YatPerformerCausalAttention

def benchmark(name, model, x, steps=10):
    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(steps):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / steps
    tokens = x.shape[0] * x.shape[1]
    tps = tokens / avg_time
    print(f"{name}: {avg_time*1000:.2f} ms | TPS: {tps:,.0f}")
    return tps

def run_benchmarks():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    B = 4
    H = 12
    D = 768
    T = 1024
    
    x = torch.randn(B, T, D, device='cuda')
    
    print(f"Benchmarking with B={B}, T={T}, D={D}, H={H}...")
    
    models = [
        ("Standard", StandardCausalAttention(D, H).cuda()),
        ("Cosformer", CosformerCausalAttention(D, H).cuda()),
        ("YatPerformer", YatPerformerCausalAttention(D, H).cuda())
    ]
    
    results = []
    for name, model in models:
        tps = benchmark(name, model, x)
        results.append({"Model": name, "TPS": tps})
        
    print("\nResults:")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_benchmarks()
