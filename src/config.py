"""
Default configuration for TinyGPT training.
"""

DEFAULT_CONFIG = {
    'context_len': 1024,
    'embed_dim': 768,
    'n_layers': 12,
    'n_heads': 12,
    'lr': 3e-4,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'weight_decay': 0.01,
    'activation_type': 'novel',
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
    'eval_interval': 500,
    'save_interval': 1000,
    'total_steps': 10000,
    'wandb_project': 'tiny-gpt-novel',
    'num_rff_features': 64,
    'num_prf_features': 16,
    'num_poly_features': 8,
    'num_quadrature_nodes': 2,
    'epsilon': 0.01,
}
