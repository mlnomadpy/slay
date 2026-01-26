"""
Data loading utilities for training.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class FineWebStream(IterableDataset):
    """Streaming dataset for FineWeb corpus."""
    
    def __init__(self, context_len=512):
        self.context_len = context_len
        self.ds = load_dataset(
            "HuggingFaceFW/fineweb",
            split="train",
            streaming=True
        )

        # Load tokenizer ONCE to read vocab size
        tok = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = tok.vocab_size
        del tok

    def __iter__(self):
        # ✅ tokenizer must be local to the worker/process
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e9)

        try:
            import deepspeed
            if deepspeed.comm.is_initialized():
                rank = deepspeed.comm.get_rank()
                world_size = deepspeed.comm.get_world_size()
            else:
                rank = 0
                world_size = 1
        except ImportError:
            rank = 0
            world_size = 1

        buffer = []

        for i, sample in enumerate(self.ds):
            if i % world_size != rank:
                continue

            tokens = tokenizer.encode(
                sample["text"],
                add_special_tokens=False
            )

            buffer.extend(tokens)

            while len(buffer) >= self.context_len + 1:
                chunk = buffer[: self.context_len + 1]
                buffer = buffer[self.context_len :]  # ✅ correct stride

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def get_eval_loader(context_len, batch_size):
    """Create evaluation dataloader from WikiText-2."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Using a small subset for speed in demo; use larger split for paper
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation") 
    
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    data = enc.input_ids[0]
    
    n_samples = (data.size(0) - 1) // context_len
    x_list, y_list = [], []
    for i in range(0, n_samples * context_len, context_len):
        x_list.append(data[i : i + context_len])
        y_list.append(data[i + 1 : i + 1 + context_len])
        
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    dataset = torch.utils.data.TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
