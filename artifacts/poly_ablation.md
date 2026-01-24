# Polynomial Approximation Ablation (Spherical Yat)

Config: B=4, T=128, C=64, H=4, ε=1e-06, R=2, M=8, P=8, device=cuda, dtype=float32

| Method | Rel L2 ↓ | MSE ↓ | Cos ↑ | Latency (ms) ↓ | Peak MB (CUDA) |
|---|---:|---:|---:|---:|---:|
| Exact (Spherical) | 0.0000 | 0.000e+00 | 1.0000 | 2.47 | 14.1 |
| Laplace-only | 0.5824 | 6.974e-03 | 0.8130 | 1.50 | 63.0 |
| Anchor | 0.6529 | 8.767e-03 | 0.7626 | 3.13 | 79.3 |
| Hadamard (shared ω) | 0.8194 | 1.381e-02 | 0.6963 | 2.26 | 63.0 |
| Nyström | 7.7650 | 1.240e+00 | 0.0296 | 6.43 | 79.3 |
| TensorSketch | 599190.3750 | 7.383e+09 | -0.0032 | 3.91 | 79.3 |
| Random Maclaurin | 2149894.7500 | 9.504e+10 | -0.0199 | 4.22 | 79.3 |
