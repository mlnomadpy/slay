"""
Shared utilities for SLAY visualization scripts.

Provides common functions for data logging, styling, and publication settings.
"""

import numpy as np
import os
from datetime import datetime


# ============================================================================
# Data Logging
# ============================================================================
def log_data(filename, data_dict, description=""):
    """
    Log plot data to a text file for LLM analysis.
    
    Args:
        filename: Output txt file path
        data_dict: Dictionary of arrays/values to log
        description: Optional description of the data
    
    Returns:
        Path to the created log file
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'assets', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# SLAY Visualization Data Log\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Description: {description}\n")
        f.write(f"#" + "="*60 + "\n\n")
        
        for key, value in data_dict.items():
            f.write(f"## {key}\n")
            if isinstance(value, np.ndarray):
                f.write(f"# Shape: {value.shape}, dtype: {value.dtype}\n")
                if value.ndim == 1 and len(value) <= 100:
                    f.write(f"# Values: {value.tolist()}\n")
                elif value.ndim == 1:
                    f.write(f"# First 20: {value[:20].tolist()}\n")
                    f.write(f"# Last 20: {value[-20:].tolist()}\n")
                    f.write(f"# Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}, Std: {value.std():.6f}\n")
                else:
                    f.write(f"# Min: {value.min():.6f}, Max: {value.max():.6f}, Mean: {value.mean():.6f}\n")
            elif hasattr(value, 'numpy'):  # torch tensor
                arr = value.detach().cpu().numpy()
                f.write(f"# Shape: {arr.shape}\n")
                f.write(f"# Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"# Length: {len(value)}\n")
                if len(value) > 50:
                    f.write(f"# Values (first 50): {list(value[:50])}...\n")
                else:
                    f.write(f"# Values: {list(value)}\n")
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, np.ndarray):
                        f.write(f"#   {sub_key}: shape={sub_val.shape}, min={sub_val.min():.4f}, max={sub_val.max():.4f}\n")
                    elif hasattr(sub_val, 'numpy'):
                        arr = sub_val.detach().cpu().numpy()
                        f.write(f"#   {sub_key}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}\n")
                    else:
                        f.write(f"#   {sub_key}: {sub_val}\n")
            else:
                f.write(f"# Value: {value}\n")
            f.write("\n")
    
    return filename


def log_csv_data(filename, columns_dict, description=""):
    """
    Log plot data as CSV for easy import.
    
    Args:
        filename: Output csv file path
        columns_dict: Dictionary where keys are column names, values are arrays
        description: Optional description
    
    Returns:
        Path to created CSV file
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else 'assets', exist_ok=True)
    
    # Convert to numpy arrays
    arrays = {}
    for key, value in columns_dict.items():
        if hasattr(value, 'numpy'):
            arrays[key] = value.detach().cpu().numpy().flatten()
        elif isinstance(value, np.ndarray):
            arrays[key] = value.flatten()
        else:
            arrays[key] = np.array(value).flatten()
    
    # Check lengths match
    lengths = [len(v) for v in arrays.values()]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        arrays = {k: v[:min_len] for k, v in arrays.items()}
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {description}\n")
        f.write(",".join(arrays.keys()) + "\n")
        for i in range(len(list(arrays.values())[0])):
            row = [f"{arrays[k][i]:.8g}" for k in arrays.keys()]
            f.write(",".join(row) + "\n")
    
    return filename
