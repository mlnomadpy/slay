# Polynomial Approximation Sweep (Spherical Yat)

Base: device=cuda, dtype=float32, embed_dim=64, heads=4, batch_size=8, eps=1e-06

## Small (T=128, R=2, M=8, P=8)

| Method | Rel L2 ↓ | Latency (ms) ↓ | Cos ↑ |
|---|---:|---:|---:|
| Exact (Spherical) | 0.0000 | 3.12 | 1.0000 |
| Laplace-only | 0.5870 | 2.78 | 0.8096 |
| Anchor | 0.6626 | 3.82 | 0.7554 |
| Hadamard (shared ω) | 0.8237 | 3.34 | 0.6940 |
| Nyström | 22.9072 | 3.41 | 0.0212 |
| TensorSketch | 474075.1562 | 5.17 | -0.0004 |
| Random Maclaurin | 2195912.7500 | 5.59 | -0.0064 |

## Medium (T=256, R=2, M=16, P=16)

| Method | Rel L2 ↓ | Latency (ms) ↓ | Cos ↑ |
|---|---:|---:|---:|
| Exact (Spherical) | 0.0000 | 0.79 | 1.0000 |
| Laplace-only | 0.5417 | 16.10 | 0.8408 |
| Anchor | 0.5667 | 18.54 | 0.8246 |
| Hadamard (shared ω) | 0.6609 | 17.44 | 0.8005 |
| Nyström | 61.6529 | 18.46 | -0.0054 |
| TensorSketch | 214115.9844 | 19.01 | 0.0135 |
| Random Maclaurin | 1715766.8750 | 18.80 | -0.0008 |

## Large (T=512, R=2, M=32, P=32)

| Method | Rel L2 ↓ | Latency (ms) ↓ | Cos ↑ |
|---|---:|---:|---:|
| Exact (Spherical) | 0.0000 | 5.02 | 1.0000 |
| Laplace-only | 0.4850 | 1905.80 | 0.8749 |
| Anchor | 0.4939 | 489.42 | 0.8695 |
| Hadamard (shared ω) | 0.6793 | 1932.07 | 0.8046 |
| Nyström | 28.1970 | 569.64 | 0.0117 |
| TensorSketch | 461739.0312 | 547.76 | 0.0023 |
| Random Maclaurin | 1772757.5000 | 551.43 | 0.0133 |

