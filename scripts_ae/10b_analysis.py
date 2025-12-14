#!/usr/bin/env python3
"""
Parse routing token files and extract token counts
Output CSV: rank, method, model_name, layer, iter, token_num
"""

import os
import re
import glob
import sys
import pandas as pd

def parse_token_file(filepath):
    """Parse single token file and extract token numbers"""
    tokens = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    line = line.strip()
                    if line.startswith('tensor(') and line.endswith(')'):
                        token_num = int(line[7:-1])
                        tokens.append(token_num)
        return tokens
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def parse_filename(filename):
    """Extract config, method, layer, rank from filename"""
    # Format: token_num_log_2_FLEX_0_0.txt
    # config_num, solver_type, layer_num, rank_num
    pattern = r'(.+)_token_counts_(.+)_(.+)_batch(\d+)_seq(\d+)_aux(.+)_layer(\d+)_(\d+)\.txt'
    match = re.match(pattern, filename)
    
    if match:
        method = match.group(1)
        model_name = match.group(2)
        layer = int(match.group(7))
        rank = int(match.group(8))
        return method, layer, rank, model_name
    return None

def find_routing_files(directories):
    """Find all routing token files in given directories"""
    all_files = []
    for directory in directories:
        if os.path.isdir(directory):
            token_files = glob.glob(os.path.join(directory, "*.txt"))
            all_files.extend(token_files)
            print(f"Found {len(token_files)} token files in {directory}")
        elif os.path.isfile(directory):
            all_files.append(directory)
    return all_files

def process_routing_files(filepaths):
    """Process routing files and return records"""
    records = []
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        
        if not parsed:
            print(f"Warning: Could not parse filename {filename}, skipping")
            continue
        
        method, layer, rank, model_name = parsed
        
        # Parse token data
        tokens = parse_token_file(filepath)
        
        if not tokens:
            continue
        
        # Create record for each iteration
        for iter_idx, token_num in enumerate(tokens):
            records.append({
                'rank': rank,
                'method': method,
                'model_name': model_name,
                'layer': layer,
                'iter': iter_idx,
                'token_num': token_num
            })
    
    return records

def main():
    """Main function"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Default routing directories
    default_dirs = [
        os.path.join(BASE_DIR, 'LAER-MoE', 'galvatron', 'models', 'moe', 'training_log', 'token_counts'),
    ]
    
    # Get directories from command line or use default
    directories = default_dirs
    
    # Default output CSV
    output_csv = os.path.join(BASE_DIR, 'LAER-MoE', 'scripts_ae', 'token_analysis.csv')
    
    print("Starting to parse routing files...")
    
    # Find routing files
    filepaths = find_routing_files(directories)
    
    if not filepaths:
        print("No routing files found")
        return
    
    # Process files
    records = process_routing_files(filepaths)
    
    if not records:
        print("No valid routing data found")
        return
    
    # Save to CSV
    df = pd.DataFrame(records)
    df = df.sort_values(['model_name', 'method', 'layer', 'rank', 'iter'])
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(records)} records to {output_csv}")
    print(f"Models: {df['model_name'].unique().tolist()}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Layers: {sorted(df['layer'].unique().tolist())}")
    print(f"Ranks: {sorted(df['rank'].unique().tolist())}")
    print(f"Iterations: {df['iter'].min()} - {df['iter'].max()}")

if __name__ == "__main__":
    main()

