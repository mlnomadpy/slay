"""
Utility functions for training and evaluation.
"""

import os
import csv
import math
import torch


def evaluate(model_engine, eval_loader):
    """Evaluate model on validation set."""
    model_engine.eval()
    total_loss = 0.0
    total_steps = 0
    device = model_engine.device
    
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            loss = model_engine(x, y)
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
    model_engine.train()
    return avg_loss, ppl


def log_metrics(log_file, step, train_loss, val_loss, ppl, tps):
    """Log training metrics to CSV file."""
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['step', 'train_loss', 'val_loss', 'perplexity', 'tokens_per_sec'])
        writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{ppl:.4f}", f"{tps:.2f}"])


class LossPlateauDetector:
    """Detects when validation loss plateaus."""
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def check(self, val_loss):
        """
        Check if loss has plateaued.
        Returns True if plateau detected, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
