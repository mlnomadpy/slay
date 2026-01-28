import argparse
import os
import torch

def convert_checkpoint(checkpoint_dir, output_file):
    """
    Convert a DeepSpeed checkpoint folder to a single fp32 pytorch file.
    """
    print(f"Converting DeepSpeed checkpoint: {checkpoint_dir} -> {output_file}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
        return

    # Use DeepSpeed's utility
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
    
    # This function loads the weights and returns the full state dict
    # It automatically handles the zero stages
    try:
        state_dict = convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_file)
        
        # DeepSpeed utility normally writes to file if output_file is provided.
        # But let's verify if we need to manually save wrapper info.
        # usually convert_zero_checkpoint_to_fp32_state_dict saves it directly if tag is not None
        # Actually, looking at DS docs, the signature is (checkpoint_dir, output_file) and it saves it.
        
        print("Conversion complete.")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Ensure you have deepspeed installed: pip install deepspeed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Path to the DeepSpeed checkpoint folder (e.g. checkpoints/step_1000)")
    parser.add_argument("output_file", help="Path to save the output .pt file")
    args = parser.parse_args()
    
    convert_checkpoint(args.checkpoint_dir, args.output_file)
