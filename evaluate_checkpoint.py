import argparse
import os
import sys
import torch
import json
import torch.nn.functional as F
from tqdm import tqdm

try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Error: lm_eval not installed. Please install it with `pip install lm_eval`")
    sys.exit(1)

from transformers import AutoTokenizer

# Adjust path to import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import TinyGPT, DEFAULT_CONFIG, ATTENTION_CLASSES

@register_model("slay-gpt")
class SlayEvalWrapper(LM):
    def __init__(self, checkpoint_path, device="cuda", batch_size=None):
        super().__init__()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size_per_gpu = batch_size or 1

        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Note: If this is a DeepSpeed checkpoint folder, you must convert it to a single .pt file first.")
            print("Usage: python deepspeed_to_fp32.py <checkpoint_dir> <output_file.pt>")
            sys.exit(1)

        # 1. Recover Configuration
        # DeepSpeed checkpoints wrapped in client_state usually look like {'config': ..., 'step': ...}
        # But the actual weights might be in 'module' or 'state_dict' or just the dict itself
        if 'config' in checkpoint:
            self.train_config = checkpoint['config']
            print("Recovered config from checkpoint.")
        else:
            print("Warning: No config found in checkpoint. Using default config from src.")
            self.train_config = DEFAULT_CONFIG

        # 2. Initialize Model
        # We need vocab size. Data script gets it from tokenizer, so we do same.
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Ensure pad token exists for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Initializing TinyGPT with attention={self.train_config['attention_type']}...")
        self.model = TinyGPT(
            vocab_size=self.tokenizer.vocab_size,
            config=self.train_config,
            attention_type=self.train_config['attention_type'],
            freeze_embeddings=False # We load trained weights anyway
        )

        # 3. Load Weights
        state_dict = checkpoint
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'client_state' in checkpoint:
            # Sometimes saved deeply nested, but standard PyTorch save is usually direct or in 'model'
            pass
        
        # Handle "module." prefix from DDP/DeepSpeed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Filter out extraneous keys if any (e.g. if config was at top level)
        model_keys = set(self.model.state_dict().keys())
        final_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
        
        missing, unexpected = self.model.load_state_dict(final_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys: {len(missing)}")
        if unexpected:
            print(f"Warning: Unexpected keys: {len(unexpected)}")
        
        self.model.to(self._device)
        self.model.eval()

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests, desc="Evaluating loglikelihood"):
            # Minimal unbatched implementation for correctness first
            # 1. Tokenize
            # Note: lm_eval might pass strings. 
            # We assume requests are (context_str, continuation_str)
            
            # Full sequence: context + continuation
            full_text = context + continuation
            full_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"].to(self._device)
            ctx_ids = self.tokenizer(context, return_tensors="pt")["input_ids"].to(self._device)
            
            # Mask out context for loss calculation
            # We only care about prob of continuation GIVEN context
            continuation_len = full_ids.shape[1] - ctx_ids.shape[1]
            
            if continuation_len <= 0:
                res.append((0.0, True))
                continue
                
            with torch.no_grad():
                # Forward pass
                # logits: [1, seq_len, vocab_size]
                logits = self.model(full_ids) 
                
                # Shift logits and labels for next-token prediction
                # logits[i] predicts ids[i+1]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = full_ids[..., 1:].contiguous()
                
                # We want the loss only on the continuation part
                # The continuation starts at index len(ctx_ids) in the full sequence
                # In shifted space (0-indexed), this corresponds to index len(ctx_ids)-1
                start_match_idx = ctx_ids.shape[1] - 1
                
                target_logits = shift_logits[0, start_match_idx:] # [cont_len, vocab]
                target_ids = shift_labels[0, start_match_idx:]    # [cont_len]
                
                # Calculate log probs
                # F.cross_entropy returns -log_prob
                # We want sum of log_probs
                loss = F.cross_entropy(target_logits, target_ids, reduction='sum')
                
                log_prob = -loss.item()
                is_greedy = (target_logits.argmax(dim=-1) == target_ids).all().item()
                
                res.append((log_prob, is_greedy))
                
        return res

    def loglikelihood_rolling(self, requests):
        return [] # Optional

    def generate_until(self, requests):
        # Placeholder / Not implemented for basic benchmarking 
        # (Needed for tasks like GSM8K, but not HellaSwag/LAMBADA-loglikelihood)
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--tasks", default="hellaswag", help="Comma-separated list of tasks")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--output_path", default="eval_results.json", help="Path to save results")
    args = parser.parse_args()

    model_wrapper = SlayEvalWrapper(args.checkpoint, device=args.device)

    task_list = args.tasks.split(",")
    results = lm_eval.simple_evaluate(
        model=model_wrapper,
        tasks=task_list,
    )

    print("\nResults:")
    print(json.dumps(results["results"], indent=2))
    
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
