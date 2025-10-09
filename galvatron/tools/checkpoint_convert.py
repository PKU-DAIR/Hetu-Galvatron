import torch
import os
import argparse
from collections import defaultdict

def convert_checkpoints_gpt(input_checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_checkpoint_path):
        if filename.endswith('.bin'):
            file_path = os.path.join(input_checkpoint_path, filename)
            checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
            layer_params = defaultdict(dict)
            for key, value in checkpoint.items():
                if len(key.split('.')) > 3:
                    layer_name = '.'.join(key.split('.')[:3])
                    key_name = '.'.join(key.split('.')[3:])
                    layer_params[layer_name][key_name] = value
                elif key.split('.')[1] == 'ln_f':
                    layer_name = '.'.join(key.split('.')[:2])
                    key_name = '.'.join(key.split('.')[2:])
                    layer_params[layer_name][key_name] = value
                else:
                    layer_name = 'transformer.embedding'
                    key_name = '.'.join(key.split('.')[1:])
                    layer_params[layer_name][key_name] = value

            for layer_name, params in layer_params.items():
                layer_file = os.path.join(output_dir, f"{layer_name.replace('.', '_')}.pt")
                torch.save(params, layer_file)
                print(f"Saved parameters for {layer_name} to {layer_file}")

def convert_checkpoints_llama(input_checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_checkpoint_path):
        if filename.endswith('.bin'):
            file_path = os.path.join(input_checkpoint_path, filename)
            try:
                checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
            except Exception as e:
                print(f"Waring: Cant read file {file_path}: {e}")
                continue
            layer_params = defaultdict(dict)
            for key, value in checkpoint.items():
                if len(key.split('.')) > 3:
                    layer_name = '.'.join(key.split('.')[:3])
                    key_name = '.'.join(key.split('.')[3:])
                    layer_params[layer_name][key_name] = value
                elif key.split('.')[1] == 'norm':
                    layer_name = '.'.join(key.split('.')[:2])
                    key_name = '.'.join(key.split('.')[2:])
                    layer_params[layer_name][key_name] = value
                elif key.split('.')[1] == 'embed_tokens':
                    layer_name = 'model.embed_tokens'
                    key_name = '.'.join(key.split('.')[1:])
                    layer_params[layer_name][key_name] = value
                else:
                    layer_name = 'lm_head'
                    key_name = key.split('.')[-1]
                    layer_params[layer_name][key_name] = value

            for layer_name, params in layer_params.items():
                layer_file = os.path.join(output_dir, f"{layer_name.replace('.', '_')}.pt")
                if os.path.exists(layer_file):
                    existing_params = torch.load(layer_file)
                    for key in params:
                        if key in existing_params:
                            existing_params[key] += params[key]
                        else:
                            existing_params[key] = params[key]
                else:
                    existing_params = params
                torch.save(existing_params, layer_file)
                print(f"Saved parameters for {layer_name} to {layer_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert large checkpoints to smaller checkpoints by layer.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model (e.g., transformer).")
    parser.add_argument("--input_checkpoint", type=str, required=True, help="Path to the input large checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the smaller checkpoints.")
    
    args = parser.parse_args()

    if args.model_type == 'gpt':
        convert_checkpoints_gpt(args.input_checkpoint, args.output_dir)
    elif args.model_type == 'llama':
        convert_checkpoints_llama(args.input_checkpoint, args.output_dir)
if __name__ == "__main__":
    main()
